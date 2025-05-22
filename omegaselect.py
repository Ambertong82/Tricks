import pandas as pd

def find_sign_changes_and_values(file1_path, file2_path, file3_path, output_path):
    # 读取第一个CSV文件
    try:
        df1 = pd.read_csv(file1_path, header=None)
    except Exception as e:
        print(f"Error reading first CSV file: {e}")
        return
    
    if len(df1) < 3:
        print("First CSV file needs at least 3 rows of data")
        return
    
    time_row = df1.iloc[0]  # 第一行是时间
    x_positions = df1.iloc[1]  # 第二行是x位置
    omega_data = df1.iloc[2:]  # 第三行开始是omega数据
    
    results = []
    
    # 遍历每一列
    for col_idx in omega_data.columns:
        omega_series = omega_data[col_idx]
        sign_changes = []
        prev_sign = None
        
        # 记录所有符号变化点（同时保存行和列索引）
        for row_idx, val in omega_series.items():
            current_sign = 1 if val >= 0 else -1
            if prev_sign is not None and current_sign != prev_sign:
                sign_changes.append({
                    'row_idx': row_idx,  # 行索引（从2开始的实际行号）
                    'col_idx': col_idx,  # 列索引
                    'sign_change': f"{'-+' if prev_sign < 0 else '+-'}",
                    'value': val
                })
            prev_sign = current_sign
        
        # 先找符合条件的负→正点
        valid_neg_to_pos = None
        for sc in sign_changes:
            if sc['sign_change'] == '-+':
                # 检查后续5行是否都为正（同一列）
                next_rows = range(sc['row_idx']+1, min(sc['row_idx']+6, omega_data.index.max()+1))
                if all(omega_data.loc[r, sc['col_idx']] >= 0 for r in next_rows):
                    valid_neg_to_pos = sc
                    break
        
        # 如果找到负→正点，再在它之后找正→负点（同一列）
        if valid_neg_to_pos:
            # 只考虑同一列且在负→正点之后的正→负点
            pos_to_neg_candidates = [sc for sc in sign_changes 
                                   if sc['sign_change'] == '+-' 
                                   and sc['col_idx'] == valid_neg_to_pos['col_idx']
                                   and sc['row_idx'] > valid_neg_to_pos['row_idx']]
            
            if pos_to_neg_candidates:
                pos_to_neg = pos_to_neg_candidates[0]  # 取第一个符合条件的
                
                results.append({
                    'column_index': valid_neg_to_pos['col_idx'],  # 列索引
                    'neg_to_pos_row': valid_neg_to_pos['row_idx'],  # 负→正行号
                    'pos_to_neg_row': pos_to_neg['row_idx'],  # 正→负行号
                    'neg_to_pos_value': valid_neg_to_pos['value'],
                    'pos_to_neg_value': pos_to_neg['value'],
                    'time': time_row[valid_neg_to_pos['col_idx']],  # 对应列的时间
                    'x_position': x_positions[valid_neg_to_pos['col_idx']]  # 对应列的x位置
                })
    
    if not results:
        print("No valid sign changes found")
        return
    
    # 读取第二个CSV文件
    try:
        df2 = pd.read_csv(file2_path, header=None)
    except Exception as e:
        print(f"Error reading second CSV file: {e}")
        return
    
    try:
        df3 = pd.read_csv(file3_path, header=None)
    except Exception as e:
        print(f"Error reading third CSV file: {e}")
        return
    
    # 准备输出结果（现在包含行列信息）
    output_data = []
    for result in results:
        try:
            # 从df2提取对应行列的值
            value1 = df2.iloc[result['neg_to_pos_row'], result['column_index']] 
            value2 = df2.iloc[result['pos_to_neg_row'], result['column_index']]
            value3 = df3.iloc[result['neg_to_pos_row'], result['column_index']]
            value4 = df3.iloc[result['pos_to_neg_row'], result['column_index']]
        except IndexError:
            print(f"Warning: Position ({result['neg_to_pos_row']}, {result['column_index']}) "
                  f"or ({result['pos_to_neg_row']}, {result['column_index']}) out of bounds")
            continue
        
        output_data.append({
            'Column_Index': result['column_index'],
            'Time': result['time'],
            'X_Position': result['x_position'],
            'Neg_to_Pos_Row': result['neg_to_pos_row'],
            'Pos_to_Neg_Row': result['pos_to_neg_row'],
            'Neg_to_Pos_Value': result['neg_to_pos_value'],
            'Pos_to_Neg_Value': result['pos_to_neg_value'],
            'yb': value1,
            'yt': value2,
            'ua_NegPos': value3,
            'ua_PosNeg': value4,
            'delta_Ua': value3 - value4,
            'Lw': value2 - value1,
            #'Next_5_Points_Valid': True,
            #'Order_Valid': result['pos_to_neg_row'] > result['neg_to_pos_row']
        })
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 查找 Time 等于 20 的 yb 值
    time_20_rows = output_df[output_df['Time'] == 20]
    if not time_20_rows.empty:
        print("Rows where Time == 35:")
        print(time_20_rows[['Time', 'yb', 'delta_Ua', 'Lw']])
    else:
        print("No rows found where Time == 20")

# 使用示例
if __name__ == "__main__":
    file1_path = "/home/amber/postpro/case090429_5_x712xomegaz.csv"  # 替换为你的第一个CSV文件路径
    file2_path = "/home/amber/postpro/case090429_5_x712xyy.csv"  # 替换为你的第二个CSV文件路径
    file3_path = "/home/amber/postpro/case090429_5_x712xub.csv"  # 替换为你的第三个CSV文件路径
    output_path = "/home/amber/postpro/mixinglayer/output_results_09_712.csv"  # 替换为你的输出CSV文件路径
    find_sign_changes_and_values(file1_path, file2_path, file3_path, output_path)

# # 使用示例
# if __name__ == "__main__":
#     file1_path = "/home/amber/postpro/case230427_4_x14xomegaz.csv"  # 替换为你的第一个CSV文件路径
#     file2_path = "/home/amber/postpro/case230427_4_x14xyy.csv"  # 替换为你的第二个CSV文件路径
#     file3_path = "/home/amber/postpro/case230427_4_x14xub.csv"  # 替换为你的第三个CSV文件路径
#     output_path = "/home/amber/postpro/mixinglayer/output_results_23_14.csv"  # 替换为你的输出CSV文件路径
#     find_sign_changes_and_values(file1_path, file2_path, file3_path, output_path)