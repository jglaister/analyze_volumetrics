import os
import os.path
import glob

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pyplot as plt
import matplotlib

class BrainVolumetrics():
    def __init__(self, rootdir, subjlist=None):
        if subjlist is None:
            search_path = os.path.join(rootdir, '*','*','*_macruise_volumes.csv')
            filelist = sorted(glob.glob(search_path))
            # TODO: Populate subjlist from this
        else:
            filelist = []
            for patient_scan in subjlist:
                patient_id, scan_id = patient_scan.split('_', 1)
                search_path = os.path.join(rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*macruise_volumes.csv')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    filelist.append(search_file[0])
                else:
                    print('No file found in ' + search_path)

        self.rootdir = rootdir
        self.subjlist = subjlist
        self.filelist = filelist
        self.brain_volumetric_df = None

    def compute_volumetrics(self):
        all_df = []
        for f in self.filelist:
            df = pd.read_csv(f, sep=',', header=0, index_col=1)
            filename = f.split('/')[-1]
            patient_id = filename.split('_')[0]
            scan_id = filename.split('_')[1]
            scan_name = patient_id + '_' + scan_id
            vol = df.transpose()[1:3]
            new_df = vol.rename(index={'Volume':scan_name, 'Name':'scan'})

            #ICV Mask
            icv_file = os.path.join(os.path.split(f)[0],patient_id + '_' + scan_id + '_MPRAGEPre_reg_mask.nii.gz')
            if os.path.exists(icv_file):
                icv = nib.load(icv_file).get_fdata()
                icv_vol = np.sum(icv)*0.8*0.8*0.8 #TODO: Read voxel size from image
                new_df['ICV'] = icv_vol
            else:
                print('No ICV in ' + filename)
                new_df['ICV'] = 'N/A'

            #Lesion
            lesion_file = os.path.join(os.path.split(f)[0],patient_id + '_' + scan_id + '_MPRAGEPre_reg_durastrip_s3dl_lesions.nii.gz')
            if os.path.exists(lesion_file):
                lesion = nib.load(lesion_file).get_fdata()
                lesion_vol = np.sum(lesion)*0.8*0.8*0.8 #TODO: Read voxel size from image
                new_df['White Matter Lesion'] = lesion_vol
            else:
                print('No lesion segmentation in ' + filename)
                new_df['White Matter Lesion'] = 'N/A'
            all_df.append(new_df)

        merged_df = pd.concat(all_df, sort=False)

        cols = merged_df.columns.tolist()
        merged_df['Cerebral White Matter Volume'] = merged_df['Right Cerebral White Matter'] + merged_df['Left Cerebral White Matter']
        merged_df['Cortical Grey Matter Volume'] = merged_df.iloc[:, cols.index('Right ACgG  anterior cingulate gyrus'):cols.index('Left TTG   transverse temporal gyrus')+1].sum(axis=1)
        merged_df['Ventricular Volume'] = merged_df['3rd Ventricle'] + merged_df['Right Inf Lat Vent'] + merged_df['Left Inf Lat Vent'] + merged_df['Right Lateral Ventricle'] + merged_df['Left Lateral Ventricle']
        merged_df['Brain Volume'] = merged_df.iloc[:, cols.index('3rd Ventricle'):cols.index('Left TTG   transverse temporal gyrus')+1].sum(axis=1)-merged_df['Ventricular Volume']

        #Reorder columns
        cols = merged_df.columns.tolist()
        cols = cols[cols.index('ICV'):] + cols[:cols.index('ICV')]
        merged_df = merged_df[cols]

        # Fill N/A with 0 - Is this correct?
        merged_df.fillna(0)
        self.brain_volumetric_df = merged_df

    def write_images(self, output_dir):
        if self.subjlist is not None:
            for patient_scan in self.subjlist:
                patient_id, scan_id = patient_scan.split('_', 1)
                search_path = os.path.join(self.rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*MPRAGEPre_reg.nii.gz')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    brain_file = search_file[0]
                else:
                    print('No file found in ' + search_path)
                    break

                search_path = os.path.join(self.rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*MPRAGEPre_reg_mask.nii.gz')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    mask_file = search_file[0]
                else:
                    print('No file found in ' + search_path)
                    break

                search_path = os.path.join(self.rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*MPRAGEPre_reg_macruise.nii.gz')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    segmentation_file = search_file[0]
                else:
                    print('No file found in ' + search_path)
                    break

                brain_data = nib.load(brain_file).get_fdata()
                mask_data = nib.load(mask_file).get_fdata()
                segmentation_data = nib.load(segmentation_file).get_fdata()
                center_slice = brain_data.shape[2]//2

                fig, axs = plt.subplots()
                axs.imshow(brain_data[:, :, center_slice].T, cmap='gray')
                axs.axis('off')
                axs.imshow(mask_data[:, :, center_slice].T, cmap='jet', alpha=0.5)
                plt.gca().set_axis_off()
                plt.margins(0, 0)
                outfile_name = os.path.join(output_dir, patient_scan + '_mask_overlay.png')
                fig.savefig(outfile_name, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                fig, axs = plt.subplots()
                axs.imshow(brain_data[:, :, center_slice].T, cmap='gray')
                axs.axis('off')
                axs.imshow(segmentation_data[:, :, center_slice].T, cmap='jet', alpha=0.5)
                plt.gca().set_axis_off()
                plt.margins(0, 0)
                outfile_name = os.path.join(output_dir, patient_scan + '_seg_overlay.png')
                fig.savefig(outfile_name, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)


    def write_volumetrics(self, output_dir, output_prefix):
        if self.brain_volumetric_df is not None:
            output_file = os.path.join(output_dir, output_prefix + '_MACRUISE_VOL.csv')
            self.brain_volumetric_df.to_csv(output_file)
