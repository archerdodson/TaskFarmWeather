import matplotlib.pyplot as plt
import torch
# import and set up the typeguard
from typeguard.importhook import install_import_hook

try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    pass

# comment these out when deploying:
install_import_hook('src.nn')
install_import_hook('src.scoring_rules')
install_import_hook('src.utils')
install_import_hook('src.parsers')
install_import_hook('src.calibration')
install_import_hook('src.weatherbench_utils')

from src.nn import ConditionalGenerativeModel, DiscardWindowSizeDim, UNet2D, DiscardNumberGenerationsInOutput
from src.utils import load_net
from src.parsers import parser_plot_weatherbench, setup
from src.weatherbench_utils import load_weatherbench_data, convert_tensor_to_da, plot_map_ax

model = 'WeatherBench'
method = 'SR'
scoring_rule = 'SignatureKernel'
kernel = 'RBFtest'  ##??
patched = False
base_measure = 'normal'
root_folder = 'results'         # Where results are stored
model_folder = 'nets'           # Subfolder for models
datasets_folder = 'results/lorenz96/datasets/'
nets_folder = 'results/nets/'
weatherbench_data_folder = "../geopotential_500_5.625deg"
weatherbench_small = False

#name_postfix = '_mytrainedmodelEnergyScore' ##Change this
name_postfix = '_mytrainedmodelSignatureKernel' ##Change this
training_ensemble_size = 3  #3/10
prediction_ensemble_size = 3 ##3/10
prediction_length = 2  
ensemble_size = prediction_ensemble_size
unet_noise_method = 'sum'  # or 'concat', etc., if relevant
unet_large = True

lr = 0.1
lr_c = 0.0
batch_size = 1
no_early_stop = True
critic_steps_every_generator_step = 1

save_plots = True
nonlinearity = 'leaky_relu'
data_size = torch.Size([10, 32, 64])              # For Lorenz63, typically data_size=1 or 3
auxiliary_var_size = 1
seed = 0

plot_start_timestep = 0
plot_end_timestep = 100

gamma = None
gamma_patched = None
patch_size = 16
no_RNN = False
hidden_size_rnn = 32

save_pdf = True

save_pdf = True

compute_patched = model in ["lorenz96", ]

model_is_weatherbench = model == "WeatherBench"

nn_model = "unet" if model_is_weatherbench else ("fcnn" if no_RNN else "rnn")

method_is_gan = False

save_pdf = True

# notice this assumes the WeatherBench dataset is considered in the daily setup.
unet_depths = (32, 64, 128, 256)

# datasets_folder, nets_folder, data_size, auxiliary_var_size, name_postfix, unet_depths, patch_size, method_is_gan, hidden_size_rnn = \
#     setup("WeatherBench", root_folder, model_folder, datasets_folder, data_size, method, scoring_rule, kernel, patched,
#           patch_size, training_ensemble_size, auxiliary_var_size, critic_steps_every_generator_step, base_measure, lr,
#           lr_c, batch_size, no_early_stop, unet_noise_method, unet_large, "unet", None)

string = f"Plot WeatherBench results with {method}"
if not method_is_gan and not method == "regression":
    string += f" using {scoring_rule} scoring rule"
print(string)

cuda = True
load_all_data_GPU = True
dataset_train, dataset_val, dataset_test = load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU,
                                                                  return_test=True,
                                                                  weatherbench_small=weatherbench_small)
print("Validation set size:", len(dataset_val))
print("Test set size:", len(dataset_test))

if method == "regression":
    net_class = UNet2D
    unet_kwargs = {"in_channels": data_size[0], "out_channels": 1,
                   "noise_method": "no noise", "conv_depths": unet_depths}
    net = DiscardWindowSizeDim(net_class(**unet_kwargs))
    net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardNumberGenerationsInOutput, net).net
else:  # SR and GAN
    # create generative net:
    inner_net = UNet2D(in_channels=data_size[0], out_channels=1, noise_method=unet_noise_method,
                       number_generations_per_forward_call=prediction_ensemble_size, conv_depths=unet_depths)
    if unet_noise_method in ["sum", "concat"]:
        # here we overwrite the auxiliary_var_size above, as there is a precise constraint
        downsampling_factor, n_channels = inner_net.calculate_downsampling_factor()
        if weatherbench_small:
            auxiliary_var_size = torch.Size(
                [n_channels, 16 // downsampling_factor, 16 // downsampling_factor])
        else:
            auxiliary_var_size = torch.Size(
                [n_channels, data_size[1] // downsampling_factor, data_size[2] // downsampling_factor])

        net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
                       size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                       number_generations_per_forward_call=prediction_ensemble_size, seed=seed + 1)
    elif unet_noise_method == "dropout":
        net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardWindowSizeDim, inner_net)

if cuda:
    net.cuda()

date = "2018-05-16"
# predict for a given date and create the plot
with torch.no_grad():
    # obtain the target and context for the specified timestring
    timestring = date + "T12:00:00.000000000"
    context, realization = dataset_test.select_time(timestring)
    print(context.shape)
    print(realization.shape)
    

    print(breakd)
    # predict the realization with the context:
    prediction = net(context.unsqueeze(1)).cpu()  # should specify how many we want

    # compute mean and standard deviation of the predictions:
    prediction_mean = prediction[0].mean(dim=0)
    prediction_std = prediction[0].std(dim=0)
    da_prediction_mean = convert_tensor_to_da(prediction_mean, realization)
    da_prediction_std = convert_tensor_to_da(prediction_std, realization)

    # prediction is shape ("batch_size", "number_generations", "height", "width", "fields"). Batch size should be 1.

    # convert to an xarray DataArray:
    da_predictions = []
    for i in range(prediction.shape[1]):
        da_predictions.append(convert_tensor_to_da(prediction[0, i], realization))

    if save_plots:
        global_projection = False
        # we do plots with 5 predictions if not deterministic
        if method == "regression":
            n_predictions_for_plots = 1
            kwargs_subplots = dict(ncols=2, nrows=1, figsize=(16 * 2.0 / 3, 2.5),
                                   subplot_kw=dict(projection=ccrs.PlateCarree(), facecolor="gray"))
        else:
            n_predictions_for_plots = 5
            kwargs_subplots = dict(ncols=3, nrows=2, figsize=(16, 4.5),
                                   subplot_kw=dict(projection=ccrs.PlateCarree(), facecolor="gray"))

        # --- plot the absolute values ---
        fig, axes = plt.subplots(**kwargs_subplots)

        # need to find max and min values over all graphs to have coherent colorbars.
        vmax = max([prediction.max().detach().numpy(), realization.max()])
        vmin = min([prediction.min().detach().numpy(), realization.min()])

        # plot both the realization and the prediction:
        p_real = plot_map_ax(realization[:, :, 0], title="Realization", ax=axes.flatten()[0],
                             global_projection=global_projection, vmax=vmax, vmin=vmin)
        for i in range(n_predictions_for_plots):
            p_pred = plot_map_ax(da_predictions[i][:, :, 0],
                                 title=f"Prediction" + ("{i + 1}" if method != "regression" else ""),
                                 ax=axes.flatten()[i + 1], global_projection=global_projection, vmax=vmax, vmin=vmin)
        # add now the colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(p_pred, cax=cbar_ax)
        fig.suptitle("Z500, " + date, size=20)

        plt.savefig(nets_folder + f"map_absolute{name_postfix}." + ("pdf" if save_pdf else "png"))

        # --- plot the differences from the realization ---
        differences = [da_predictions[i] - realization for i in range(n_predictions_for_plots)]

        fig, axes = plt.subplots(**kwargs_subplots)
        # need to find max and min values over all graphs to have coherent colorbars.
        vmax = max([differences[i].max() for i in range(n_predictions_for_plots)])
        vmin = min([differences[i].min() for i in range(n_predictions_for_plots)])

        for i in range(n_predictions_for_plots):
            p_pred = plot_map_ax(differences[i][:, :, 0],
                                 title=f"Prediction" + ("{i + 1}" if method != "regression" else ""),
                                 ax=axes.flatten()[i + 1], global_projection=global_projection, vmax=vmax, vmin=vmin)
        # add now the colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(p_pred, cax=cbar_ax)
        fig.suptitle("Z500, predictions - realization, " + date, size=20)

        plt.savefig(nets_folder + f"map_differences{name_postfix}." + ("pdf" if save_pdf else "png"))

        if method != "regression":
            # --- plot the differences with respect to ensemble mean ---
            differences = [da_predictions[i] - da_prediction_mean for i in range(n_predictions_for_plots)]
            realization_diff = realization - da_prediction_mean

            fig, axes = plt.subplots(**kwargs_subplots)
            # need to find max and min values over all graphs to have coherent colorbars.
            vmax = max([differences[i].max() for i in range(n_predictions_for_plots)])
            vmin = min([differences[i].min() for i in range(n_predictions_for_plots)])

            p_real = plot_map_ax(realization_diff[:, :, 0], title="Realization", ax=axes[0, 0],
                                 global_projection=global_projection, vmax=vmax, vmin=vmin)
            for i in range(n_predictions_for_plots):
                p_pred = plot_map_ax(differences[i][:, :, 0], title=f"Prediction {i + 1}",
                                     ax=axes.flatten()[i + 1], global_projection=global_projection, vmax=vmax,
                                     vmin=vmin)
            # add now the colorbar:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(p_pred, cax=cbar_ax)
            fig.suptitle("Z500, Centered in mean prediction, " + date, size=20)

            plt.savefig(nets_folder + f"map_differences_ens_mean{name_postfix}." + ("pdf" if save_pdf else "png"))

            # --- plot the ensemble mean and std ---

            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16 * 2.0 / 3, 3),
                                     subplot_kw=dict(projection=ccrs.PlateCarree(), facecolor="gray"))

            p_real = plot_map_ax(da_prediction_mean[:, :, 0], title="Mean", ax=axes[0],
                                 global_projection=global_projection)
            p_pred = plot_map_ax(da_prediction_std[:, :, 0], title=f"Standard deviation",
                                 ax=axes[1], global_projection=global_projection)

            fig.suptitle("Z500, Prediction mean and standard deviation, " + date, size=20)

            plt.savefig(nets_folder + f"map_differences_mean_std{name_postfix}." + ("pdf" if save_pdf else "png"))
