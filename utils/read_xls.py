import pandas as pd

import os

g = os.walk('../data/dx_xls/dx_xls2')
f1all_name = '../data/dx_xls/all1.dat'
f2all_name = '../data/dx_xls/all2.dat'
idx = 4861
for path, dir_list, file_list in g:
    for file_name in file_list:
        try:
            f_path = os.path.join(path, file_name)
            df = pd.read_excel(f_path, sheet_name='result')
            f1_name = '../data/dx_xls/dx_txt/{0}.sp1'.format(idx)
            f2_name = '../data/dx_xls/dx_txt/{0}.sp2'.format(idx)
            sp1_all = ''
            sp2_all = ''
            # print(df['对话人'], df['识别结果'])
            print(idx)

            for index, row in df.iterrows():
                if row['对话人'] == 'speaker1':
                    sp1_all += ';' + row['识别结果']
                    with open(f1_name, 'a+') as f_out:
                        f_out.writelines(row['识别结果'] + '\n')
                elif row['对话人'] == 'speaker2':
                    sp2_all += ';' + row['识别结果']
                    with open(f2_name, 'a+') as f_out:
                        f_out.writelines(row['识别结果'] + '\n')

            with open(f1all_name, 'a+') as f_out:
                f_out.writelines(sp1_all + '\n')
            with open(f2all_name, 'a+') as f_out:
                f_out.writelines(sp2_all + '\n')

            idx += 1

        except:
            pass
