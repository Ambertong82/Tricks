import pandas as pd
file_list = [
    '/home/amber/postpro/rawdata/case230427_4_{}.csv'.format(i) for i in range(0, 79, 1)]
# file_list=['alpha_1.csv','alpha_2.csv']
all_results = pd.DataFrame()
data_dict = []
data_dict2 = []
for file in file_list:
    df = pd.read_csv(file)
    #filtered_df=df[(df['alpha.a']>1e-4)&(df['alpha.a']<0.001)]
    #filtered_df = df[(df['alpha.saline'] > 1e-5) & (df['Points:1'] > 0)]
    filtered_df=df[(df['alpha.a']>5e-5) & (df['Points:1'] > 0)]
    if len(filtered_df) > 1:
        max_point = filtered_df['Points:0'].idxmax()
        #result = filtered_df.loc[max_point, [
            #'Time', 'Points:0', 'Points:1', 'alpha.saline', 'U:0']]
        result = filtered_df.loc[max_point, [
            'Time', 'Points:0', 'Points:1', 'alpha.a', 'U.a:0']]
        # time=filtered_df.loc[max_point,'Time']
        # print(time)
        all_results = pd.concat([all_results, result.to_frame().T])
all_results.to_csv('/home/amber/postpro/case230427_4_5e5.csv', index=False)
