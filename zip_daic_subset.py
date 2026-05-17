import os
import zipfile

def create_daic_subset_zip(src_dir, output_zip):
    print(f"Creating zip file: {output_zip}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Root files (.xlsx, .pdf, .csv)
        for f in os.listdir(src_dir):
            file_path = os.path.join(src_dir, f)
            if os.path.isfile(file_path):
                # Ignore temp excel files (~$)
                if (f.endswith('.xlsx') or f.endswith('.pdf') or f.endswith('.csv')) and not f.startswith('~$'):
                    zipf.write(file_path, f)
                    print(f"Added root file: {f}")
        
        # 2. 'documents' and 'util' folders
        for folder_name in ['documents', 'util']:
            folder_path = os.path.join(src_dir, folder_name)
            if os.path.exists(folder_path):
                for root, _, files in os.walk(folder_path):
                    for f in files:
                        full_path = os.path.join(root, f)
                        arcname = os.path.relpath(full_path, src_dir)
                        zipf.write(full_path, arcname)
                        print(f"Added: {arcname}")
                        
        # 3. Patient folders 300_P to 330_P
        for f in os.listdir(src_dir):
            folder_path = os.path.join(src_dir, f)
            if os.path.isdir(folder_path) and f.endswith('_P'):
                try:
                    patient_id = int(f.split('_')[0])
                except ValueError:
                    continue
                    
                # Filter range 300 - 330
                if 300 <= patient_id <= 330:
                    for p_file in os.listdir(folder_path):
                        # Hanya ambil audio (.wav) dan csv, kecualikan yang mengandung 'CLNF'
                        if 'CLNF' not in p_file and (p_file.endswith('.wav') or p_file.endswith('.csv')):
                            full_path = os.path.join(folder_path, p_file)
                            arcname = os.path.join(f, p_file) # Simpan dengan struktur foldernya (misal: 300_P/300_AUDIO.wav)
                            zipf.write(full_path, arcname)
                            print(f"Added: {arcname}")

if __name__ == "__main__":
    # Path disesuaikan dengan struktur repo kamu
    src = r"d:\repositories\menthealth-ai\data\raw\DAIC-WOZ"
    out = r"d:\repositories\menthealth-ai\DAIC-WOZ_300-330_Subset.zip"
    
    if os.path.exists(src):
        create_daic_subset_zip(src, out)
        print(f"\nSelesai! File zip berhasil disimpan di: {out}")
    else:
        print(f"Error: Folder {src} tidak ditemukan.")
