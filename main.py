import tensorflow as tf
import logging
import sys
import time
from utils import main_utils
from functionalities import main_functions, model_utils

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

def run_mmtscnet():
    """
    Executes program to preprocess data aswell as tune and train the classification network.
    """
    args = main_utils.parse_arguments()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    main_utils.setup_logging(log_level)
    logging.info("Starting MMTSCNET - This may take a while!")
    data_dir, work_dir, model_dir, elim_per, max_pcscale, sss_test, cap_sel, grow_sel, bsize, img_size, pc_size = main_utils.validate_inputs(args.datadir, args.workdir, args.modeldir, args.elimper, args.maxpcscale, args.ssstest, args.capsel, args.growsel, args.batchsize, args.numpoints)
    fwf_av = main_utils.are_fwf_pointclouds_available(data_dir)

    if args.prediction == False:
        logging.info("=== MMTSCNET TRAINING MODE ===")
        try:
            logging.info("Creating Working environment...")
            workspace_paths = main_functions.extract_data(data_dir, work_dir, fwf_av, cap_sel, grow_sel)

            logging.info("Preprocessing data...")
            X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict = main_functions.preprocess_data(workspace_paths, sss_test, cap_sel, grow_sel, elim_per, max_pcscale, pc_size, img_size, fwf_av)

            logging.info("Commencing hyperparameter-tuning...")
            untrained_model = main_functions.perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, pc_size, img_size, num_classes, cap_sel, grow_sel)

            logging.info("Training MMTSCNet...")
            trained_model = main_functions.perform_training(untrained_model, bsize, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, model_dir, label_dict, cap_sel, grow_sel)

            logging.info("Training finished, you can now predict for %s_%s data. User --prediction to create a new prediction!", cap_sel, grow_sel)

        except Exception as e:
            #Log error if program does not execute correctly
            logging.exception("An error occurred: %s", e)
            time.sleep(3)
            sys.exit(1)
    
    else:
        logging.info("=== MMTSCNET PREDICTION MODE ===")
        try:
            logging.info("Loading pretrained model...")
            if main_utils.check_if_model_is_created(model_dir) == True:
                pretrained_model_path = model_utils.get_trained_model_folder(model_dir, cap_sel, grow_sel)
                pretrained_model = model_utils.load_trained_model_from_folder(pretrained_model_path)

                logging.info("Predicting for custom dataset...")
                main_functions.predict_for_custom_data(pretrained_model, work_dir, img_size, pc_size, cap_sel, grow_sel, elim_per, fwf_av, data_dir, model_dir)

            else:
                logging.error("No pretrained model available, exiting!")
                sys.exit(3)

        except Exception as e:
            #Log error if program does not execute correctly
            logging.exception("An error occurred: %s", e)
            time.sleep(3)
            sys.exit(1)


if __name__ == "__main__":
    run_mmtscnet()