# iDCF
This is a pytorch implementation of the paper: Debiasing Recommendation by Learning Identifiable Latent Confounders  published at SIGKDD 2023.

## Environment Requirement
The code has been tested running under Python 3.8.10 The required packages are as follows:
* pytorch == 1.13.0
* numpy == 1.22.3
* pandas == 1.4.2
* ray[tune] == 2.4.0
* bottleneck == 1.3.7
* protobuf == 3.19.0

## Dataset
* [Coat](https://www.cs.cornell.edu/schnabts/mnar/)
* [Yahoo! R3](https://webscope.sandbox.yahoo.com/)
* [KuaiRand](https://kuairand.com/)
* Synthetic

## How to run the code
Take Coat as an example
1. Build the dataset via `build_dataset.ipynb`
2. Train the ivae model to learn the latent confounder (add ``--tune `` for searching hyperparameters)

    ``python3 ivae_exposure.py  --dataset coat  --patience 100 ``
3. Save confounder models via `save_ae_params.ipynb`
4. Run the feedback prediction model (add ``--tune `` for searching hyperparameters), we have uploaded the confounder models, one can directly run the following code (similar for other baselines):

    ``
    python3 iDCF.py --topk 5  --dataset coat --patience 20 
    ``

## Acknowledgment
Some codes are adopted from  
* https://github.com/Dingseewhole/Robust_Deconfounder_master
* https://github.com/AIflowerQ/InvPref_KDD_2022
* https://github.com/siamakz/iVAE

Thanks for their contributions!