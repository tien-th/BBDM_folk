import os


def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def get_image_paths(dataset_path, stage):
    stage_path = os.path.join(dataset_path, f'{stage}')
    
    image_paths_ct = []
    image_paths_pet = []
    
    for patient_id in os.listdir(stage_path):
        patient_folder = os.path.join(stage_path, patient_id)
        
        if os.path.isdir(patient_folder):
            ct_file = os.path.join(patient_folder, 'ct.npy')
            pet_file = os.path.join(patient_folder, 'pet.npy')
            
            if os.path.exists(ct_file):
                image_paths_ct.append(ct_file)
            if os.path.exists(pet_file):
                image_paths_pet.append(pet_file)

    return image_paths_ct, image_paths_pet