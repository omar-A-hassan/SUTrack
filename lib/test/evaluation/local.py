from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set Kaggle working paths here.
    settings.prj_dir = '/kaggle/working/SUTrack'
    settings.result_plot_path = '/kaggle/working/SUTrack/test/result_plots'
    settings.results_path = '/kaggle/working/SUTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/kaggle/input/datasets/potatoguy44/teacher/sutrack_l384'    # Root path for model weights
    settings.network_path = '/kaggle/working/SUTrack/test/networks'    # Where tracking networks are stored.
    settings.segmentation_path = '/kaggle/working/SUTrack/test/segmentation_results'

    # Kaggle Dataset paths (Adjust the dataset names mapped in /kaggle/input/)
    settings.uav_path = '/kaggle/input/datasets/hanaramah/uav123-20l-dataset/UAV123'
    
    # Optional datasets left blank
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.lasotlang_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.otblang_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

