/idr_generation/
├── aa_index_table.pkl
    this is a dictionary that maps each amino acid 1-letter rep to an int between 0 and 19
    you may want to use this to convert (200k, 2) str sequences to (200k, 2, 20) int sequences 

├── analysis
│   ├── compare_uniprot.py
    analysis script for comparing generated sequences to random segments sampled from uniprot
    python evaluate_vs_uniprot.py \
        --checkpoint esm2_fes_finetuned.pt \
        --uniprot    uniprot.fasta \
        [--n_samples  256]
        [--temperature 1.0]
        [--device     cuda]

│   ├── eval_uncond_gen.py
    compares the distribution of real and generated sequences from unconditional model

│   └── get_distributions.py
    plots different amino acid frequency distributions.

├── bestm
│   ├── part1top_p10_2048_1e-3.pth
│   ├── part2top_p10_2048_1e-3.pth
│   ├── part3top_p10_2048_1e-3.pth
│   ├── part4top_p10_2048_1e-3.pth
│   └── part5top_p10_2048_1e-3.pth
├── concs_grid.npy
├── encoder_decoder_model.pt
    Model weights for the conditional generative model

├── encoder_decoder_model.py
    architecture of conditional generative model

├── example_fe_validation.py
├── fe_predictor_tf_linc_meanz.py
├── generate_fe_landscapes.py
├── generate.py
    Samples sequences from the conditional generative model

├── helpers.py
    various helpers for loading data and models

├── README.txt
├── temperature_scaling.py
    runs a sweep over sampling temperatures and generates sequences unconditionally

├── train_encoder_decoder_clust_testset.out
    output of training conditional generative model with cluster train/test split

├── train_encoder_decoder.py
    script for training conditional generative model.

├── train_encoder_decoder_rand_testset.out
    output of training conditional generative model with random train/test split

├── train_unconditional.out
    output of training unconditional model.

├── train_unconditional.py
    training script for unconditional generative model

├── unconditional_decoder.pt
    model weights for 

└── unconditional_model.py
    unconditional generative model architecture

concs_grid.npy: (1326, 2)
The grided concentration pairs that free energy is evaluated for
unit is chain/nm^3
the grid size is 0.005, with c1+c2 < 0.25
for each pair of sequences, free energy is evaluated at these 1326 grid points, so the free energy surface is represented as a (1326,) vector

unilat_seqs.npy: (200k, 2)
the pairs of sequences of the data, sequence length is 20

unilat_fes.npy: (5, 200k, 1326) 
this is the free energy landscape for the 200k pairs of sequences evaluated at 5 cross-validated models
to build the model, you only need the mean of the 5 models, unilat_fes.mean(axis = 0), which will give you an arr of (200k, 1326)
however, unilat_fes.std(axis = 0) can provide you information of the uncertainty of the model at each free energy evaluation, in case this is needed

aa_index_table.pkl
this is a dictionary that maps each amino acid 1-letter rep to an int between 0 and 19
you may want to use this to convert (200k, 2) str sequences to (200k, 2, 20) int sequences 

The goal:
input: (Batch size = N, 1326): N free energy surfaces
output: (N, 2) str arr or (N, 2, 20) int arr, N pairs of sequences

To test the prediction, use generate_fe_landscapes.py
Input 1: (N, 2, 20) int arr of seqs that you generated
Input 2: output filename, the file would be a (5, N, 1326) arr that you can compare with your input
example: example_fe_validation.py
