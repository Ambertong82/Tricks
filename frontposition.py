import pandas as pd
file_list = [
    '/home/amber/postpro/rawdata/case0704_2Bo_{}.csv'.format(i) for i in range(0, 29, 1)]
# file_list=['alpha_1.csv','alpha_2.csv']
all_results = pd.DataFrame()
data_dict = []
data_dict2 = []
for file in file_list:
    df = pd.read_csv(file)
    # filtered_df=df[(df['alpha.a']>0.0005)&(df['alpha.a']<0.001)]
    filtered_df = df[(df['alpha.saline'] > 1e-5) & (df['Points:1'] > 0)]
    # filtered_df=df[(df['alpha.a']>0.00001)]
    if len(filtered_df) > 1:
        max_point = filtered_df['Points:0'].idxmax()
        result = filtered_df.loc[max_point, [
            'Time', 'Points:0', 'Points:1', 'alpha.saline', 'U:0']]
        # time=filtered_df.loc[max_point,'Time']
        # print(time)
        all_results = pd.concat([all_results, result.to_frame().T])
all_results.to_csv('/home/amber/postpro/case0704_2Bo_1e5.csv', index=False)
