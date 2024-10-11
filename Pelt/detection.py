import os.path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import ruptures as rpt
import scipy
from ruptures.exceptions import BadSegmentationParameters
import json
import pyabf 
class NotFittable(Exception):
    pass


def CUSUM(input, delta, h, verbose=False):
    """
    Function used in the detection of abrupt changes in mean current; optimal for Gaussian signals.
    CUSUM is based on the cummulative sum algorithm.
    This function will define new start and end points more precisely than just
    the RecursiveLowPassFast and will fit levels inside the TransolocationEvent objects.

    Parameters
    ----------
    input : numpy array
        Input signal.
    delta : float
        Most likely jump to be detected in the signal.
    h : float
        Threshold for the detection test.

    Returns
    -------
    mc : the piecewise constant segmented signal
    kd : a list of float detection times (in samples)
    krmv : a list of float estimated change times (in samples).
    """

    # initialization
    Nd = k0 = 0
    kd = []
    krmv = []
    k = 1
    l = len(input)
    m = np.zeros(l)
    m[k0] = input[k0]
    v = np.zeros(l)
    sp = np.zeros(l)
    Sp = np.zeros(l)
    gp = np.zeros(l)
    sn = np.zeros(l)
    Sn = np.zeros(l)
    gn = np.zeros(l)

    while k < l:
        m[k] = np.mean(input[k0:k + 1])
        v[k] = np.var(input[k0:k + 1])

        sp[k] = delta / v[k] * (input[k] - m[k] - delta / 2)
        sn[k] = -delta / v[k] * (input[k] - m[k] + delta / 2)

        Sp[k] = Sp[k - 1] + sp[k]
        Sn[k] = Sn[k - 1] + sn[k]

        gp[k] = np.max([gp[k - 1] + sp[k], 0])
        gn[k] = np.max([gn[k - 1] + sn[k], 0])

        if gp[k] > h or gn[k] > h:
            kd.append(k)
            if gp[k] > h:
                kmin = np.argmin(Sp[k0:k + 1])
                krmv.append(kmin + k0)
            else:
                kmin = np.argmin(Sn[k0:k + 1])
                krmv.append(kmin + k0)

            # Re-initialize
            k0 = k
            m[k0] = input[k0]
            v[k0] = sp[k0] = Sp[k0] = gp[k0] = sn[k0] = Sn[k0] = gn[k0] = 0

            Nd = Nd + 1
        k += 1
    if verbose:
        print('delta:' + str(delta))
        print('h:' + str(h))
        print('Nd: ' + str(Nd))
        print('krmv: ' + str(krmv))

    if Nd == 0:
        mc = np.mean(input) * np.ones(k)
    elif Nd == 1:
        mc = np.append(m[krmv[0]] * np.ones(krmv[0]), m[k - 1] * np.ones(k - krmv[0]))
    else:
        mc = m[krmv[0]] * np.ones(krmv[0])
        for ii in range(1, Nd):
            mc = np.append(mc, m[krmv[ii]] * np.ones(krmv[ii] - krmv[ii - 1]))
        mc = np.append(mc, m[k - 1] * np.ones(k - krmv[Nd - 1]))
    return (mc, kd, krmv)


def RecursiveLowPassFast(signal, samplerate, rough_detec):
    """
    Function used to find roughly where events are in a noisy signal using a first order recursive
    low pass filter defined as :
        u[k] = a*u[k-1]+(1-a)*i[k]
        with u the mean value at sample k, i the input signal and a < 1, a parameter.
    """
    s = rough_detec.get("s", 5)
    e = rough_detec.get("e", 0)
    a = rough_detec.get("a", 0.999)
    max_event_length = rough_detec.get("max_event_length", 5e-1)


    signal = np.ravel(signal)

    padlen = np.uint64(samplerate)
    prepadded = np.ones(padlen) * np.mean(signal[0:1000])
    signaltofilter = np.concatenate((prepadded, signal))

    mltemp = scipy.signal.lfilter([1 - a, 0], [1, -a], signaltofilter)
    vltemp = scipy.signal.lfilter([1 - a, 0], [1, -a], np.square(signaltofilter - mltemp))

    ml = np.delete(mltemp, np.arange(padlen, dtype=int))
    vl = np.delete(vltemp, np.arange(padlen, dtype=int))


    sl = ml - s * np.sqrt(vl)


    Ni = len(signal)
    points = np.array(np.where(signal <= sl)[0])
    to_pop = np.array([], dtype=int)
    for i in range(1, len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop = np.append(to_pop, i)
    points = np.unique(np.delete(points, to_pop))
    RoughEventLocations = []
    NumberOfEvents = 0

    for i in points:
        if NumberOfEvents != 0:
            if i >= RoughEventLocations[NumberOfEvents - 1][0] and i <= RoughEventLocations[NumberOfEvents - 1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] + e * np.sqrt(vl[i])
        Mm = ml[i]
        Vv = vl[i]
        duration = 0
        endp = start
        if (endp + 1) < len(signal):
            while signal[endp + 1] < El and endp < (Ni - 2):
                duration += 1
                endp += 1
        if duration >= max_event_length * samplerate or endp > (
                Ni - 10):
            NumberOfEvents -= 1
            continue
        else:
            k = start
            while signal[k] < Mm and k > 1:
                k -= 1
            start = k - 1
            k2 = i + 1
            # while signal[k2] > Mm:
            #    k2 -= 1
            # endp = k2
            if start < 0:
                start = 0
            RoughEventLocations.append((start, endp, np.sqrt(vl[start]), ml[start]))

    return np.array(RoughEventLocations)



def find_rough_event_loc(signal, samplerate, rough_detec):
    """
    Parameters
    ----------
    signal       signal to analyse. If signal too long, memory capacity might not suffice, since calculating std need lots of memory
    samplerate
    rough_detec: dictionary with rough detection parameters. Here "dt" referes to the window size for calculating
                 the sliding average/variance, the higher the value the slower.

    Returns
    -------

    """

    dt = rough_detec.get("dt_exact", 100)
    s = rough_detec.get("s", 5)
    e = rough_detec.get("e", 0)
    max_event_length = rough_detec.get("max_event_length", 5e-1)
    lag = rough_detec.get("lag", 2)

    event_start_indices = []
    event_end_indices = []
    event_local_std = []
    event_local_baseline = []

    if isinstance(signal, np.ndarray):
        signal = np.ravel(signal)


    window = sliding_window_view(signal, window_shape=(dt,))
    local_baseline = np.mean(window, axis=1)
    local_std = np.std(window, axis=1)


    event_start_detected = False
    for i in range(dt, len(signal)):

        if not event_start_detected:
            event_start_threshold = local_baseline[i-dt-1] - s * local_std[i-dt-1-lag]
            event_end_threshold = local_baseline[i-dt-1] - e * local_std[i-dt-1-lag]

        if not event_start_detected and signal[i] < event_start_threshold and signal[i - 1] >= event_start_threshold:
            event_start_indices.append(i)
            event_local_baseline.append(local_baseline[i-dt])
            event_local_std.append(local_std[i-dt])
            event_start_detected = True

        if event_start_detected:
            if signal[i] > event_end_threshold and signal[i - 1] <= event_end_threshold:
                event_end_indices.append(i)
                event_start_detected = False


    event_rough_infos = []
    for i in range(len(event_start_indices)):
        try:
            if (event_end_indices[i] - event_start_indices[i]) >= max_event_length * samplerate: continue
            event_rough_infos.append((event_start_indices[i], event_end_indices[i], event_local_std[i], event_local_baseline[i]))
        except IndexError:
            continue

    return event_rough_infos




class Events:
    def __init__(self, ev_infos, samplerate, signal, fit_params, dt_baseline=50,
                 fit_method="pelt", show=False):
        self.dt_baseline = dt_baseline
        self.ev_infos = ev_infos
        self.samplerate = samplerate
        self.signal = signal
        self.fit_method = fit_method
        self.fit_params = fit_params
        self.show = show
        self.events = self.create_events()

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.events):
            raise StopIteration
        value = self.events[self.i]
        self.i += 1
        return value

    def create_events(self):
        events = []
        fit_event_thresh = self.fit_params.get("fit_event_thresh", 10)
        fit_level_thresh = self.fit_params.get("fit_level_thresh", 7)
        for ev_info in self.ev_infos:
            ev_start = int(ev_info[0])
            ev_end = int(ev_info[1])
            local_std = ev_info[2]
            local_baseline = ev_info[3]
            dt_baseline = self.dt_baseline
            signal = np.array(self.signal[ev_start - dt_baseline: ev_end + dt_baseline])
            try:

                event_to_add = Event(signal, local_baseline=local_baseline, local_std=local_std,
                                           ev_indices=(ev_start, ev_end), fit_method=self.fit_method, fit_params=self.fit_params,
                                           samplerate=self.samplerate, fit_level_thresh=fit_level_thresh,
                                           fit_event_thresh=fit_event_thresh, show=self.show, dt_baseline=dt_baseline)

                if self.show:
                    signal = event_to_add.corrected_signal
                    time = np.linspace(0, len(signal) / self.samplerate, len(signal))
                    print(event_to_add.lvls_info)
                    plt.plot(time, signal)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Current (nA)')
                    for lvl_info in event_to_add.lvls_info:
                        plt.axhline(lvl_info[0], color="red")
                    plt.axvline(time[event_to_add.ev_start], color="green")
                    plt.axvline(time[event_to_add.ev_end], color="green")
                    plt.title('Event plot')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

            except IndexError:
                continue
            except NotFittable:
                continue
            except BadSegmentationParameters:
                continue
            else:
                events.append(event_to_add)

        return events



class Event:
    def __init__(self, signal, local_baseline, local_std, ev_indices, fit_params,
                 samplerate, fit_level_thresh, fit_event_thresh, dt_baseline,
                 fit_method="c_pelt", show=False):
        self.signal = signal
        self.local_std = local_std
        self.local_baseline = local_baseline
        self.dt_baseline = dt_baseline
        self.ev_indices = ev_indices
        self.fit_method = fit_method
        self.fit_params = fit_params
        self.samplerate = samplerate
        self.fit_level_thresh = fit_level_thresh
        self.fit_event_thresh = fit_event_thresh
        self.show = show
        if self.fit_method == "cusum":
            self.ct = self.cusum_lvl_fit()
        elif self.fit_method == "pelt":
            self.ct = self.pelt()
        elif self.fit_method == "dynamic":
            self.ct = self.dyn()
        elif self.fit_method == "c_dynamic":
            self.ct = self.c_dyn()
        elif self.fit_method == "c_pelt":
            self.ct = self.c_pelt()
        self.set_feat()
        self.nr_lvls = len(self.lvls_info)
        if self.nr_lvls == 1:
            self.height = -1 * self.lvls_info[0][0]
        else:
            self.height = self.get_height()

    def cusum_lvl_fit(self):
        hbook = self.fit_params.get("hbook")
        delta = self.fit_params.get("delta")
        sigma = self.fit_params.get("sigma")
        if sigma:
            h = hbook * delta / sigma
        else:
            h = hbook * delta / self.local_std

        const_sig, det_t, ct = CUSUM(self.signal, delta, h)
        if ct[-1] - ct[0] < self.fit_event_thresh:
            raise NotFittable("event not fittable")

        return ct



    def pelt(self):
        pen = self.fit_params.get("pen")
        model = self.fit_params.get("model", "l2")
        min_size = self.fit_params.get("min_size_lvl", 3)

        ev = self.signal

        if pen == "BIC":
            pen = np.var(ev) * np.log(len(ev)) if model == "l2" else np.log(len(ev))
        elif pen == "AIC":
            pen = np.var(ev)

        algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(pen=pen)

        if self.show:
            print(ct)
            rpt.display(ev, ct, figsize=(10, 6))
            plt.show()

        if ct[-2] - ct[0] < self.fit_event_thresh:
            raise NotFittable("event not fittable")

        return ct[:-1]

    def c_pelt(self):
        pen = self.fit_params.get("pen")
        model = self.fit_params.get("model", "l2")
        min_size = self.fit_params.get("min_size_lvl", 3)

        ev = self.signal

        if pen == "BIC":
            pen = np.var(ev) * np.log(len(ev))
        elif pen == "AIC":
            pen = np.var(ev)

        if model == "l2":
            kernel = "linear"
        else:
            raise NotImplemented("chosen model not implemented for KernelCPD algorithm yet.")
        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(pen=pen)

        if self.show:
            print(ct)
            rpt.display(ev, ct, figsize=(10, 6))
            plt.show()

        if ct[-2] - ct[0] < self.fit_event_thresh:
            raise NotFittable("event not fittable")

        return ct[:-1]


    def dyn(self):
        nr_ct = self.fit_params.get("nr_ct")
        model = self.fit_params.get("model", "l2")
        min_size = self.fit_params.get("min_size_lvl", 3)

        ev = self.signal

        algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(n_bkps=nr_ct)

        if self.show:
            print(ct)
            rpt.show.display(ev, ct, figsize=(10, 6))
            plt.title('Change Point Detection: Dynamic Programming Search Method')
            plt.show()

        if ct[-2] - ct[0] < self.fit_event_thresh:
            raise NotFittable("event not fittable")

        return ct[:-1]


    def c_dyn(self):
        nr_ct = self.fit_params.get("nr_ct")
        model = self.fit_params.get("model", "l2")
        min_size = self.fit_params.get("min_size_lvl", 3)

        ev = self.signal

        if model == "l2":
            kernel = "linear"
        else:
            raise NotImplemented("chosen model not implemented for KernelCPD algorithm yet.")
        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=1).fit(ev)
        ct = algo.predict(n_bkps=nr_ct)

        if self.show:
            print(ct)
            rpt.display(ev, ct, figsize=(10, 6))
            plt.show()

        if ct[-2] - ct[0] < self.fit_event_thresh:
            raise NotFittable("event not fittable")

        return ct[:-1]



    def set_feat(self):
        self.baseline = np.mean(self.signal[0:self.ct[0]])
        self.ev_start = self.ct[0]
        self.ev_end = self.ct[-1]
        self.dwell = (self.ev_end - self.ev_start)/self.samplerate
        self.corrected_signal = list(map(lambda x: x-self.baseline, self.signal.ravel()))
        self.mean = np.abs(np.mean(self.corrected_signal[self.ev_start:self.ev_end+1]))
        self.std = np.std(self.corrected_signal[self.ev_start:self.ev_end+1])

        lvls_info = []
        for i in range(1, len(self.ct)):
            lvl_curr = np.mean(self.signal[self.ct[i - 1]:self.ct[i]]) - self.baseline
            lvl_dwell_sample = self.ct[i] - self.ct[i-1]
            if lvl_dwell_sample <= self.fit_level_thresh:
                continue
            else:
                lvls_info.append((lvl_curr, lvl_dwell_sample))

        if not lvls_info:
            raise NotFittable("levels too short")

        self.event_total_indices = (self.ev_indices[0] - self.dt_baseline + self.ev_start, self.ev_indices[0] - self.dt_baseline + self.ev_end)
        self.lvls_info = lvls_info



    def get_height(self):
        levels_cleaned = [lvl for lvl, _ in self.lvls_info]
        height = max(levels_cleaned) - min(levels_cleaned)
        return height





def get_events(signal, samplerate, fit_params, fit_method="pelt", rough_detec_params={}, show=False, folder_path=".", filename="result.json"):

    rd_algo = rough_detec_params.get("rd_algo")
    dt_baseline = fit_params.get("dt_baseline", 50)
    rough_detec_algorithm = find_rough_event_loc if rd_algo == "exact" else RecursiveLowPassFast


    event_info = rough_detec_algorithm(signal, samplerate, rough_detec_params)
    print("nr events (rough detec): ", len(event_info))



    fitted_events = Events(event_info, samplerate=samplerate, dt_baseline=dt_baseline,
                           signal=signal, fit_method=fit_method, fit_params=fit_params, show=show)


    events = []
    for event_to_add in fitted_events:
        if not event_to_add.height: continue
        if event_to_add.height < 0: continue

        new_ev = {
                  "signal_w_baseline": list(map(lambda x: float(x), event_to_add.corrected_signal)),
                  "start_end_in_sig": (int(event_to_add.ev_start), int(event_to_add.ev_end)),
                  "local_baseline": float(event_to_add.local_baseline),
                  "level_info": list(map(lambda x: (float(x[0]), int(x[1])), event_to_add.lvls_info)),
                  "mean": float(event_to_add.mean),
                  "start_end_in_raw": list(map(lambda x: int(x), event_to_add.event_total_indices)),
                  "std": float(event_to_add.std),
                  "change_times": list(map(lambda x: int(x), event_to_add.ct)),
                  "height": float(event_to_add.height),
                  "dwell": float(event_to_add.dwell),
                  }

        events.append(new_ev)

    results = {
        "samplerate": samplerate,
        "rough_detec_params": rough_detec_params,
        "fit_params": fit_params,
        "fit_method": fit_method,
        "events": events,
    }

    path = os.path.join(folder_path, f"{filename}.json")
    with open(path, "w") as file:
        json.dump(results, file, indent=4)

    return




# abf_file = "/work/zf267656/peltfolder/dnadata/20210703 wtAeL 4M KCl A4 p2 120mV-9.abf"
# signal = pyabf.ABF(abf_file)
# signal_data = signal.data[0] 
# sigpro = signal_data[102736: 105200]
# samplerate = 40_000_000
# rd_algo = "exact"
# rough_detec_params = {
#         "s": 1,                         #threshhold of detecting start of event
#         "e": 0,                         #threshhold of detection end of event (after start was detected)
#         # "dt_exact": 50,               #number of points used to calculate local std/mean (only use when using rd_algo : exact, but generally not neccessary)
#         "rd_algo": "exact",           #if left out, use a recursive lowpass filter for thresh hold, in general leave this out
#         # "lag": 1,                     #only necessary when rd_algo : exact is used. Leave it at 1 or 2, not that important
#         # "max_event_length": 10e-5,    #maximum duration for a signal to be considered an event and not pore clogging for example (in SI units)
#     }
# rough_detec_algorithm = find_rough_event_loc if rd_algo == "exact" else RecursiveLowPassFast
# event_info = rough_detec_algorithm(sigpro, samplerate, rough_detec_params)
# print(event_info)