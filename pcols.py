import numpy as np
from functools import partial


axis_dict = {0: 'sw', 1: 'step'}
esquared_over_h = 38.740E-6 # in Siemens.
h_over_esquared = 1 / esquared_over_h
dac_chan_names = ['CH' + str(i) for i in range(20)]
coarse_chan_names = dac_chan_names[::4] + dac_chan_names[1::4]
fine_chan_names = dac_chan_names[2::4] + dac_chan_names[3::4]
dac_fine_div = 200.0
dac_fine_offset = 0.05
name_func_dict = {
    'backgate': {'label': 'Backgate voltage (V)'},
    'MC': {'label': 'Mixing chamber temperature (K)'},
    'mL': {'label': 'Left MIX gate voltage (V)'},
    'mR': {'label': 'Right MIX gate voltage (V)'},
    'Bx': {'label': 'Magnetic field in x direction (T)'},
    'Bz': {'label': 'Magnetic field in z direction (T)'},
}

# Select which columns to calculate with the sides list.
sides = ['', 'left', 'right']
#sides = ['']

def underscore(prefix, suffix):
    if not suffix:
        s = prefix
    else:
        s = '{}_{}'.format(prefix, suffix)
    return s

def meta_for_side(meta, side):
    meta = meta['setup']['meta']
    if side:
        meta = meta[side]
    return meta

def get_dac_from_meta(meta):
    for ins in meta['register']['instruments']:
        if ins['name'] == 'dac':
            return ins

def get_dac_val(meta, dac_ch_name, chan_is_fine=False):
    if not dac_ch_name in dac_chan_names:
        msg = ('dac_ch_name {} not recognized '
               'as the name of a DAC channel.'.format(dac_ch_name))
        print(msg)
        return
    dac = get_dac_from_meta(meta)
    if chan_is_fine:
        dac_ch_num = int(dac_ch_name[2:])
        if dac_ch_name in coarse_chan_names:
            c_ch_name = dac_ch_name
            f_ch_name = dac_ch_name[:2] + str(dac_ch_num+2)
        elif dac_ch_name in fine_chan_names:
            c_ch_name = dac_ch_name[:2] + str(dac_ch_num-2)
            f_ch_name = dac_ch_name
        c_ch_val = dac['current_values'][c_ch_name]
        f_ch_val = dac['current_values'][f_ch_name]
        return c_ch_val + f_ch_val/dac_fine_div + dac_fine_offset
    else:
        return dac['current_values'][dac_ch_name]


def signal(data, pdata, meta):
    setup = meta['setup']['meta']
    for ins in meta['register']['instruments']:
        if ins['name'] == setup['lock_sig_source']:
            break
    signal_value = float(ins['config']['SLVL']) / setup['lock_sig_divider']
    column = np.full(shape=data.shape, fill_value=signal_value)
    return column

name = signal.__name__
func = signal
label = 'signal (V)'
name_func_dict[name] = {'func': func, 'label': label}


def time_delta(data, pdata, meta):
    time = data['time']
    start_time = np.take(time, indices=0)
    return time - start_time

name = time_delta.__name__
func = time_delta
label = 'Time since start (seconds)'
name_func_dict[name] = {'func': func, 'label': label}


def time_delta_min(data, pdata, meta):
    return pdata['time_delta'] / 60

name = time_delta_min.__name__
func = time_delta_min
label = 'Time since start (minutes)'
name_func_dict[name] = {'func': func, 'label': label}


def lockin_readings(data, pdata, meta, bias_curr, side):
    """
    lockin_curr lockin_bias
    """
    if bias_curr == 'curr':
        amp_name = 'current_amp'
    elif bias_curr == 'bias':
        amp_name = 'bias_amp'
    else:
        return
    lockin_name = underscore('lockin_'+bias_curr, side) + '/X'
    meta = meta_for_side(meta, side)
    gain = np.abs(meta[amp_name])
    column = data[lockin_name] / gain
    return column

for side in sides:
    for bias_curr in ('bias', 'curr'):
        name = underscore('lockin_' + bias_curr, side)
        func = partial(lockin_readings, bias_curr=bias_curr, side=side)
        if bias_curr == 'bias':
            unit = 'V'
        elif bias_curr == 'curr':
            unit = 'A'
        label = underscore('lockin', side) + ' {} ({})'.format(bias_curr, unit)
        name_func_dict[name] = {'func': func, 'label': label}


def conductance2(data, pdata, meta, side):
    current = pdata[underscore('lockin_curr', side)]
    bias = pdata['signal']
    column = current / bias / esquared_over_h
    return column

for side in sides:
    name = underscore(conductance2.__name__, side)
    func = partial(conductance2, side=side)
    label = (side + ' 2T dI/dV from AC (e^2/h)').lstrip(' ')
    name_func_dict[name] = {'func': func, 'label': label}


def conductance4(data, pdata, meta, side):
    current = pdata[underscore('lockin_curr', side)]
    bias = pdata[underscore('lockin_bias', side)]
    column = current / bias / esquared_over_h
    return column

for side in sides:
    name = underscore(conductance4.__name__, side)
    func = partial(conductance4, side=side)
    label = (side + ' 4T dI/dV from AC (e^2/h)').lstrip(' ')
    name_func_dict[name] = {'func': func, 'label': label}


def d_resistance4(data, pdata, meta, side):
    return 1 / pdata[underscore('conductance4', side)]

for side in sides:
    name = underscore(d_resistance4.__name__, side)
    func = partial(d_resistance4, side=side)
    label = (side + ' 4T dV/dI from AC (h/e^2)').lstrip(' ')
    name_func_dict[name] = {'func': func, 'label': label}


def dc_diff_conductance4(data, pdata, meta, side):
    """
    For very small biases (small numbers in array bias) the numbers in out_arr
    will blow up. out_arr is not clipped to improve transparency and since in
    this case the interesting interval of values is always [0,4] which is easy
    to remember and input manually as limits in the browser.
    """
    curr = pdata['dc_current2']
    bias = data['dc_bias']
    dI = np.diff(curr, axis=0)
    dV = np.diff(bias, axis=0)
    out_arr = np.full(shape=curr.shape, fill_value=np.nan)
    out_arr[:-1] = dI / dV / esquared_over_h
    return out_arr

for side in sides:
    name = underscore('dc_diff_conductance4', side)
    func = partial(dc_diff_conductance4, side=side)
    label = (side + ' 4T dI/dV from DC (e^2/h)').lstrip(' ')
    name_func_dict[name] = {'func': func, 'label': label}


def dc_conductance(data, pdata, meta, side):
    m = meta_for_side(meta, side)
    curr = data[underscore('dc_curr', side)] / m['current_amp']
    bias = data[underscore('dc_bias', side)] / m['bias_amp']
    return curr / bias / esquared_over_h

for side in sides:
    name = underscore(dc_conductance.__name__, side)
    func = partial(dc_conductance, side=side)
    label = (side + ' DC conductance I/V (e^2/h)').lstrip(' ')
    name_func_dict[name] = {'func': func, 'label': label}


def dc_current2(data, pdata, meta, side):
    m = meta_for_side(meta, side)
    curr = data[underscore('dc_curr', side)] / m['current_amp']
    return curr

for side in sides:
    name = underscore(dc_current2.__name__, side)
    func = partial(dc_current2, side=side)
    label = 'DC current {} (A)'.format(side)
    name_func_dict[name] = {'func': func, 'label': label}


def dc_bias(data, pdata, meta, side):
    m = meta_for_side(meta, side)
    curr = data[underscore('dc_bias', side)] / m['bias_amp']
    return curr

for side in sides:
    name = underscore(dc_bias.__name__, side)
    func = partial(dc_bias, side=side)
    label = 'DC bias {} (V)'.format(side)
    name_func_dict[name] = {'func': func, 'label': label}


def logcond(data, pdata, meta, full_col_name):
    arr = pdata[full_col_name]
    return np.log10(np.abs(arr))

for side in sides:
    for col_name in ('conductance2', 'conductance4', 'dc_conductance',
                     'dc_current2'):
        full_col_name = underscore(col_name, side)
        name = 'log_' + full_col_name
        func = partial(logcond, full_col_name=full_col_name)
        if col_name == 'dc_current2':
            label = 'log10(I) (log10(A))'
        else:
            label = 'log10(dI/dV) (log10(e^2/h))'
        name_func_dict[name] = {'func': func, 'label': label}


def rootcond(data, pdata, meta, full_col_name):
    arr = pdata[full_col_name]
    return np.sign(arr) * np.sqrt(np.abs(arr))

for side in sides:
    for col_name in ('conductance2', 'conductance4', 'dc_conductance',
                     'dc_current2'):
        full_col_name = underscore(col_name, side)
        name = 'root_' + full_col_name
        func = partial(rootcond, full_col_name=full_col_name)
        if col_name == 'dc_current2':
            label = 'sqrt(I) (sqrt(A))'
        else:
            label = 'sqrt(dI/dV) with sign (sqrt(e^2/h))'
        name_func_dict[name] = {'func': func, 'label': label}