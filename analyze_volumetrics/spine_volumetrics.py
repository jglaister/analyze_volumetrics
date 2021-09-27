import os
import os.path
import glob

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pyplot as plt

class SpineVolumetrics():
    def __init__(self, rootdir, subjlist=None):
        if subjlist is None:
            search_path = os.path.join(rootdir, '*','*','*_macruise_volumes.csv')
            filelist = sorted(glob.glob(search_path))
            # TODO: Populate subjlist from this
        else:
            filelist = {'CSA': [], 'MTR': [], 'MTR_WMGM': []}

            for patient_scan in subjlist:
                patient_id, scan_id = patient_scan.split('_', 1)
                search_path = os.path.join(rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*SPINE_CSA_perslice.csv')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    filelist['CSA'].append(search_file[0])
                else:
                    print('No file found in ' + search_path)

            for patient_scan in subjlist:
                patient_id, scan_id = patient_scan.split('_', 1)
                search_path = os.path.join(rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*SPINE_MTR_perslice.csv')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    filelist['MTR'].append(search_file[0])
                else:
                    print('No file found in ' + search_path)

            for patient_scan in subjlist:
                patient_id, scan_id = patient_scan.split('_', 1)
                search_path = os.path.join(rootdir, patient_id, scan_id,
                                           patient_id + '_' + scan_id + '*SPINE_avg_GM_WM_MTR.csv')
                search_file = glob.glob(search_path)
                if len(search_file) > 0:
                    filelist['MTR_WMGM'].append(search_file[0])
                else:
                    print('No file found in ' + search_path)

        self.rootdir = rootdir
        self.subjlist = subjlist
        self.filelist = filelist
        self.spine_csa_df = None
        self.spine_mtr_df = None

    def compute_average_WMGM_MTR(self):
        n = 0
        GM = 0
        WM = 0
        for f in self.filelist['MTR_WMGM']:
            n = n + 1
            df = pd.read_csv(f, sep=' ')
            GM = GM + df['GM_MTR'][0]
            WM = WM + df['WM_MTR'][0]

        self.avg_GM_MTR = GM / n
        self.avg_WM_MTR = WM / n

    def compute_volumetrics(self):
        df_mtr = pd.DataFrame(columns = ['Patient_id','Avg_MTR', 'Norm_MTR'])
        for f in self.filelist['MTR']:
            df_f = pd.read_csv(f, sep=',', header=0, index_col=1)
            filename = f.split('/')[-1]
            patient_id = filename.split('_')[0]
            scan_id = filename.split('_')[1]
            scan_name = patient_id + '_' + scan_id
            avg_MTR = np.sum(df_f[3:-3,6] * df_f[3:-3,5]) / np.sum(df_f[3:-3,5])
            norm_MTR = (avg_MTR-self.avg_GM_MTR)/(self.avg_WM_MTR-self.avg_GM_MTR)
            df_mtr.append({'Patient_id': scan_name, 'Avg_MTR': avg_MTR, 'Norm_MTR': norm_MTR}, ignore_index=True)

        df_csa = pd.DataFrame(columns = ['Patient_id','Avg_CSA'])
        for f in self.filelist['CSA']:
            df_f = pd.read_csv(f, sep=',',header=0,index_col=1).to_numpy()
            filename = f.split('/')[-1]
            patient_id = filename.split('_')[0]
            scan_id = filename.split('_')[1]
            scan_name = patient_id + '_' + scan_id
            avg_CSA = np.mean(df_f[3:-3,4])
            df_csa = df_csa.append({'Patient_id': scan_name, 'Avg_CSA': avg_CSA},ignore_index=True)

        self.spine_mtr_df = df_mtr
        self.spine_csa_df = df_csa

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
        if self.spine_mtr_df is not None:
            output_file = os.path.join(output_dir, output_prefix + '_SPINE_MTR.csv')
            self.spine_mtr_df.to_csv(output_file)

        if self.spine_csa_df is not None:
            output_file = os.path.join(output_dir, output_prefix + '_SPINE_CSA.csv')
            self.spine_csa_df.to_csv(output_file)

