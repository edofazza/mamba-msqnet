import csv
import os


def create_lighter_annots(input_file, output_file):
    ovids = []
    rows = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=' ')
        for row in reader:
            ovid = row['original_vido_id']
            labels = row['labels']
            if labels == '' or ovid in ovids:
                continue
            ovids += [ovid]
            rows += [[ovid, labels]]
            #print([ovid, labels])
    header = ['video_id', 'labels']
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    directory = "/Users/edoardofazzari/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/PhD/reseaches/thesis/animalkingdom/annotation"
    input_file = os.path.join(directory, 'train.csv')
    output_file = os.path.join(directory, 'train_light.csv')
    create_lighter_annots(input_file, output_file)
    input_file = os.path.join(directory, 'val.csv')
    output_file = os.path.join(directory, 'val_light.csv')
    create_lighter_annots(input_file, output_file)
