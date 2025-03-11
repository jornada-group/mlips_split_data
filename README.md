# Supporting Data for "*Accurate, transferable, and verifiable machine-learned interatomic potentials for layered materials*"

## About the Repository
The `allegro_config_files` contains the config files that were used to train the Allegro models. The models for Figure 2 and Figure S1 were trained on the older Allegro version, and their corresponding YAML files are found in `allegro_config_files/models_fig2`. All the interlayer models were trained on the jornada_group fork of allegro found here: . The files have been modified to point to the corresponding dataset in al `allegro_config_files/datasets`. The `surrogate_models` folder contain the pth files used to produce the plots in Figures 3, 4. We share the pth here, as the weights are important in reproducing the results of the paper, to load them please ensure to be using pytorch 2.0 or later. The script used to corrupt the models can be found in `surrogate_models/corrupt_models.py`. The interlayer yaml file used to relax the HfS2/GaS bilayer is found in  `allegro_config_files/models_HfS2_GaS`

## Training your own interlayer models
We recommend that you currently use the MACE interlayer model training procedure available in the jornada_group [github](https://github.com/jornada-group/mace-interlayer/tree/akash_interlayer_modify?tab=readme-ov-file#interlayer-mace). We will release nequip and allegro versions compatible with their current develop branch shortly.


