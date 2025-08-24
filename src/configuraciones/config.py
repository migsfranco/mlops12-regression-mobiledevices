# Categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA_FREQUENT = ['']

CATEGORICAL_VARS_WITH_NA_MISSING = ['battery_time']

# Numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = []

# Temporal variables
TEMPORAL_VARS = ['weight', 'ram']

# Reference variable
REF_VAR = "price_range"

# This variable is to calculate the temporal variable, can be dropped afterwards
DROP_FEATURES = ['weight', 'ram']

# Variables to log transform
NUMERICALS_LOG_VARS = [
    'blue',
    'clock_speed',
    'dual_sim',
    'fc',
    'four_g',
    'int_memory',
    'm_dep',
    'mobile_wt',
    'n_cores',
    'pc',
    'px_height',
    'px_width',
    'ram',
    'sc_h',
    'sc_w',
    'talk_time',
    'three_g',
    'touch_screen',
    'wifi'
]

# Variables to binarize
BINARIZE_VARS = [
    'battery_power',
    'clock_speed',
    'fc',
    'int_memory',
    'px_height',
    'sc_h',
    'sc_w'
]

# Variables to map
QUAL_VARS = [
    'battery_power',
    'clock_speed',
    'fc',
    'int_memory',
    'px_height',
    'sc_h',
    'sc_w'
]

EXPOSURE_VARS = ['touch_screen']

FINISH_VARS = ['wifi']

GARAGE_VARS = ['four_g', 'three_g', 'dual_sim']

FENCE_VARS = ['mobile_wt', 'talk_time']

# Categorical variables to encode
CATEGORICAL_VARS = [
    'blue',
    'four_g',
    'three_g',
    'dual_sim',
    'touch_screen',
    'wifi',
    'battery_power',
    'int_memory',
    'mobile_wt',
    'ram',
    'talk_time',
    'px_height',
    'px_width',
    'sc_h',
    'sc_w',
    'n_cores',
    'm_dep'
]

# Variable mappings
QUAL_MAPPINGS = {'Low': 1, 'Med': 2, 'High': 3, 'Very High': 4}

EXPOSURE_MAPPINGS = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

FINISH_MAPPINGS = {'Missing': 0, 'NA': 0, 'Unf': 1,
                   'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

GARAGE_MAPPINGS = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

# The selected variables
FEATURES = [
    'battery_power',
    'blue',
    'clock_speed',
    'dual_sim',
    'fc',
    'four_g',
    'int_memory',
    'm_dep',
    'mobile_wt',
    'n_cores',
    'pc',
    'px_height',
    'px_width',
    'ram',
    'sc_h',
    'sc_w',
    'talk_time',
    'three_g',
    'touch_screen',
    'wifi',
    'weight',
    'battery_time'
]
