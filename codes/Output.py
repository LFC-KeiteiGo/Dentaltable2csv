from codes.OutputTools import *
import os


plist = os.listdir('C:/Users/neilc/Projects/Dental_Panorama/image_dataformat/output_cellparsed')
plist = [x for x in plist if 'PNum' in x]

teeth = ['LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LU1', 'LU2', 'LU3', 'LU4', 'LU5', 'LU6', 'LU7', 'LU8',
         'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RU1', 'RU2', 'RU3', 'RU4', 'RU5', 'RU6', 'RU7', 'RU8']
notations_main = ['0', '1', '2', '3', '4', '5', '6', '7']
notations_sub = ['0', '/', '1', 'x']

data = ['No', 'Pos', 'Exist', 'Caries', 'PeriodontalDisease', 'ApicalLesion']
template = pd.DataFrame(columns=data)

for patient in plist:
    pnum = int(patient[4:])
    x_m, y_m = extract_file(patient, prediction_main, dws.listm_pred)
    x_s, y_s = extract_file(patient, prediction_sub, dws.lists_pred)
    file_names = x_m + x_s
    classes = y_m + y_s

    for tooth in teeth:
        new_row = pd.DataFrame({'No': [pnum], 'Pos': tooth, 'Exist': [1],
                                'Caries': [0], 'PeriodontalDisease': [0], 'ApicalLesion': [0]})
        files_tooth = [x for x in file_names if tooth in x]
        files_sub = [x for y in ['Others', 'ApicalLesion'] for x in files_tooth if y in x]
        classes_sub = np.array([np.array(classes[file_names.index(x)]) for x in files_sub])

        # Check Exist
        if sum(classes_sub)[1] > 0.6:
            new_row['Exist'] = 0
            template = pd.concat([template, new_row], ignore_index=True)
            continue

        # Check Caries
        status_caries = [classes[file_names.index(x)] for x in files_tooth if 'Caries' in x][0]
        new_row['Caries'] = notations_main[status_caries.index(max(status_caries))]

        # Check Periodontal
        status_caries = [classes[file_names.index(x)] for x in files_tooth if 'Alveolar' in x][0]
        new_row['PeriodontalDisease'] = notations_main[status_caries.index(max(status_caries))]

        # Check Apical
        status_caries = [classes[file_names.index(x)] for x in files_tooth if 'ApicalLesion' in x][0]
        new_row['ApicalLesion'] = notations_main[status_caries.index(max(status_caries))]

        template = pd.concat([template, new_row], ignore_index=True)

table_prediction = template

# Read Annotated data
table_condition = pd.read_csv('C:/Users/neilc/Projects/Dental_Panorama/OriginalData/teacher_label.csv')

table_condition = table_condition.iloc[:6400,:]
table_conditions = table_condition.astype({'Exist': 'int',  'Caries':'int', 'PeriodontalDisease':'int', 'ApicalLesion':'int' })
table_condition.loc[table_condition['Exist'] == -1,'Exist'] = 1  # Remove埋伏

table_predictions = table_prediction.iloc[:6400,:]
table_predictions = table_predictions.astype({'Exist': 'int',  'Caries': 'int', 'PeriodontalDisease':'int', 'ApicalLesion':'int' })

print('accuracy of Existense is :{}'.format(
    sum(table_conditions['Exist'] == table_predictions['Exist'])/len(table_conditions['Exist'])))

table_conditionr = table_conditions.loc[table_conditions['Exist'] == table_predictions['Exist']]
table_predictionr = table_predictions.loc[table_predictions['Exist'] == table_conditions['Exist']]
print('accuracy of Caries is :{}'.format(
    sum(table_conditionr['Caries'] == table_predictionr['Caries'])/len(table_conditionr['Caries'])))
print('accuracy of Periodontal is :{}'.format(
    sum(table_conditionr['PeriodontalDisease'] == table_predictionr['PeriodontalDisease'])/len(table_conditionr['PeriodontalDisease'])))
print('accuracy of ApicalLesion is :{}'.format(
    sum(table_conditionr['ApicalLesion'] == table_predictionr['ApicalLesion'])/len(table_conditionr['ApicalLesion'])))


pnums = os.listdir('image_dataformat/output_cellparsed')
for pnum in pnums:
    imgs, fnames = load_singlepatient('image_dataformat/output_cellparsed', pnum)
    big_img = create_tableimage(imgs, fnames, dws, prediction_main, prediction_sub)
    cv2.imwrite('outputs/'+pnum+'.jpg', big_img)

table_prediction.to_csv('outputs/generallist.csv')