import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Global variables and settings
path = 'curves/smallBox_210.CSV'
steady_thresh = 12
p_heating = [25, 250, 50, 30]  # [initial, final, tc, offset]
p_cooling = [250, 26, 500, 0]


# Fit exponential curves to heating and cooling data
def init_fin_exp(x, initial, final, tc, offset):
    return np.piecewise(x, [x < offset, x >= offset], [initial, lambda x1: final + (initial - final) * np.exp(-(x1 - offset) / tc)])


def fit_exp(df, p):
    return curve_fit(init_fin_exp, df['Time'].values, df['Temperature(°C)'].values, p0=p, maxfev=10000,)[0]


def init_fin_exp_str(initial, final, tc, offset):
    initial = round(initial)
    final = round(final)
    tc = round(tc)
    offset = round(offset)
    x1 = rf'x-{offset}'
    fraction = rf'\frac{{{x1}}}{{{tc}}}'
    exp = rf'${final}+({initial}-{final})e^{fraction}$'
    print(exp)
    return exp


# Append a row to a dataframe
def df_append(df, row):
    df.loc[len(df.index)] = row


def main():
    # Load and condition main data
    data = pd.read_csv(path, encoding='unicode_escape')
    for col in data.columns:
        if 'EnviroPad' in col or 'Unnamed' in col:
            data.drop(col, axis=1, inplace=True)

    # convert d/m/y hour:min:sec to secs
    data['Time'] = pd.to_datetime(data['Time'], format='%d/%m/%Y %H:%M:%S')
    data['Time'] = (data['Time'] - data['Time'].min()).dt.total_seconds()

    # Separate states (heating, steady, cooling)
    states_df = {
        'heating': pd.DataFrame(columns=['Time', 'Temperature(°C)']),
        'steady': pd.DataFrame(columns=['Time', 'Temperature(°C)']),
        'cooling': pd.DataFrame(columns=['Time', 'Temperature(°C)'])
    }

    steady_temp = data['Temperature(°C)'].max() - steady_thresh
    state = 'heating'
    for i, row in data.iterrows():
        temp = row['Temperature(°C)']
        if state == 'heating':
            if temp < steady_temp:
                df_append(states_df['heating'], row)
            else:
                state = 'steady'
                df_append(states_df['steady'], row)
        elif state == 'steady':
            if temp > steady_temp:
                df_append(states_df['steady'], row)
            else:
                state = 'cooling'
                df_append(states_df['cooling'], row)
        elif state == 'cooling':
            df_append(states_df['cooling'], row)
        else:
            raise Exception(f'{state} state is not valid')

    # Set each df to start at t=0
    for key, df in states_df.items():
        df['Time'] = df['Time'] - df['Time'].min()

    # Fit an exponential curve
    # [initial, final, tc, offset]
    h_initial, h_final, h_tc, h_offset = fit_exp(states_df['heating'], p_heating)
    c_initial, c_final, c_tc, c_offset = fit_exp(states_df['cooling'], p_cooling)

    heating_x = list(states_df['heating'].index)
    heating_y = [init_fin_exp(x, h_initial, h_final, h_tc, h_offset) for x in heating_x]
    cooling_x = list(states_df['cooling'].index)
    cooling_y = [init_fin_exp(x, c_initial, c_final, c_tc, c_offset) for x in cooling_x]

    # Create a new figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns of subplots

    # Plot each DataFrame in a separate subplot
    axs[0].plot(states_df['heating']['Time'], states_df['heating']['Temperature(°C)'])
    axs[0].plot(heating_x, heating_y)
    axs[0].set_title('Heating')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Temperature(°C)')
    axs[0].set_xlim(states_df['heating'].index.min(), states_df['heating'].index.max())
    axs[0].text(0.01, 0.92, init_fin_exp_str(h_initial, h_final, h_tc, h_offset), transform=axs[0].transAxes, fontsize=12, color='red')

    axs[1].plot(states_df['steady']['Time'], states_df['steady']['Temperature(°C)'])
    axs[1].set_title('Steady')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Temperature(°C)')
    axs[1].set_xlim(states_df['steady'].index.min(), states_df['steady'].index.max())

    axs[2].plot(states_df['cooling']['Time'], states_df['cooling']['Temperature(°C)'])
    axs[2].plot(cooling_x, cooling_y)
    axs[2].set_title('Cooling')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Temperature(°C)')
    axs[2].set_xlim(states_df['cooling'].index.min(), states_df['cooling'].index.max())
    axs[2].text(0.01, 0.92, init_fin_exp_str(c_initial, c_final, c_tc, c_offset), transform=axs[2].transAxes, fontsize=12, color='red')

    # Set x-axis ticks to include only start and end times
    for ax in axs:
        ax.set_xticks([ax.get_xlim()[0], ax.get_xlim()[1]])

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5, left=0.08, right=0.98)

    # Show the figure with all three subplots
    plt.savefig('plot.jpg')
    plt.show()


if __name__ == '__main__':
    main()
