import entre
import joblib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model", type=str)
    
    args = parser.parse_args()
    
    subj_id_ls, obj_id_ls = joblib.load('to_replace_id_ls.output')
    care_id_ls = sorted(list(set(subj_id_ls + obj_id_ls)))
    # care_id_ls = list(range(12418))
    
    if args.model == 'luke':
        key_array, pred_array, pred_prob_array, id_to_label = entre.lukeInference(args.input_file, list(range(15509)))
    elif args.model == 'ire':
        key_array, pred_array, pred_prob_array, id_to_label = entre.ireInference(args.input_file, list(range(15509)))
    print('f1 score ' , entre.getF1Micro(key_array[care_id_ls], pred_array[care_id_ls]))
    
    joblib.dump((key_array, pred_array, pred_prob_array, id_to_label), args.output_file)
