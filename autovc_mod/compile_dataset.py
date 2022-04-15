import os
import shutil


loc_a = "~/Data/ts_segments_accom_wavenet/"
loc_b = "~/Data/ts_segments_vocals_wavenet/"
output_dir = "~/Data/ts_segments_combined"

loc_a_files = []
for r, d, f in os.walk(loc_a):
    for file in f:
        if '.npy' in file and 'feats' in file and 'accom' in file:
            loc_a_files.append(file)

loc_b_files = []
for r, d, f in os.walk(loc_b):
    for file in f:
        if '.npy' in file and 'feats' in file and 'vocals' in file:
            loc_b_files.append(file)

# Match files in loc_a_files and loc_b_files
print(len(loc_a_files), len(loc_b_files))
matched_files = []
for file_a in loc_a_files:
    found_match = False
    for file_b in loc_b_files:
        if file_a.split('_accom')[0] == file_b.split('_vocals')[0]:
            found_match = True
            matched_files.append((file_a, file_b))
    if not found_match:
        print("No match found for:", file_a)

print(len(matched_files))

# Save matched files in output_dir
for match in matched_files:
    file_a, file_b = match
    shutil.copy(os.path.join(loc_a, file_a), os.path.join(output_dir, file_a))
    shutil.copy(os.path.join(loc_b, file_b), os.path.join(output_dir, file_b))
