import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from tqdm import tqdm
import umap.umap_ as umap
import multiprocessing

# import data
data = pd.read_csv('../tsne_umap_tutorials/data/APOGEEDR17_GAIAEDR3_noflagfilter.csv', delimiter=',')
# data = pd.read_csv('APOGEEDR17_GAIAEDR3_noflagfilter.csv', delimiter=',')

# preprocessing
# Spatial:
ra = data["RA"]   #APOGEE
dec = data["DEC"]   #APOGEE

# Identification:
apogee_ID = data["# APOGEE_ID_"]   #APOGEE
gaia_ID = data["GAIAEDR3_SOURCE_ID"]  # Gaia

# Kinematic:
parallax = data["GAIAEDR3_PARALLAX"]  # Gaia
pmra = data["GAIAEDR3_PMRA"]  # Gaia
pmra_err = data["GAIAEDR3_PMRA_ERROR"]  # Gaia
pmdec = data["GAIAEDR3_PMDEC"]  # Gaia
pmdec_err = data["GAIAEDR3_PMDEC_ERROR"]  # Gaia
RV = data["VHELIO_AVG"]   #APOGEE
RV_err = data["VERR"]   #APOGEE
#dist = data["dist"]   #APOGEE no dist in this dataset?
#dist_err = data["dist_err"]   #APOGEE
jr = data["jr"]   #APOGEE
jr_err = data["jr_err"]   #APOGEE
jz = data["jz"]   #APOGEE
jz_err = data["jz_err"]   #APOGEE
#jphi = data["jphi"]   #APOGEE no jphi in this dataset?
#jphi_err = data["jphi_err"]    #APOGEE

# # Spectral (useful for filtering):
TEFF_ERR = data["TEFF_ERR"]   #APOGEE
TEFF = data["TEFF"]   #APOGEE
LOGG_ERR = data["LOGG_ERR"]   #APOGEE
LOGG = data["LOGG"]   #APOGEE
SNR = data["SNR"]   #APOGEE
ASPCAPFLAG = data["ASPCAPFLAG"]
STARFLAG = data["STARFLAG"]

# # Chemical abundances from astroNN:
FE_H = data['FE_H'] 
C_FE = data['C_FE']
CI_FE = data['CI_FE']
N_FE = data['N_FE']
O_FE = data['O_FE']
MG_FE = data['MG_FE']
AL_FE = data['AL_FE']
SI_FE = data['SI_FE']
P_FE = data['P_FE']
S_FE = data['S_FE']
K_FE = data['K_FE']
CA_FE = data['CA_FE']
TI_FE = data['TI_FE']
TIII_FE = data['TIII_FE']
V_FE = data['V_FE']
CR_FE = data['CR_FE']
MN_FE = data['MN_FE']
CO_FE = data['CO_FE']
NI_FE = data['NI_FE']

# # Chemical abundance errors from astroNN:
FE_H_err = data["FE_H_ERR"] 
C_FE_err = data['C_FE_ERR']
CI_FE_err = data['CI_FE_ERR']
N_FE_err = data['N_FE_ERR']
O_FE_err = data['O_FE_ERR']
MG_FE_err = data['MG_FE_ERR']
AL_FE_err = data['AL_FE_ERR']
SI_FE_err = data['SI_FE_ERR']
P_FE_err = data['P_FE_ERR']
S_FE_err = data['S_FE_ERR']
K_FE_err = data['K_FE_ERR']
CA_FE_err = data['CA_FE_ERR']
TI_FE_err = data['TI_FE_ERR']
TIII_FE_err = data['TIII_FE_ERR']
V_FE_err = data['V_FE_ERR']
CR_FE_err = data['CR_FE_ERR']
MN_FE_err = data['MN_FE_ERR']
CO_FE_err = data['CO_FE_ERR']
NI_FE_err = data['NI_FE_ERR']

# # Number of stars in the initial sample of APOGEE DR16: 
print("There are {} stars in our initial sample".format(len(ra)))

cols = [ra, dec, apogee_ID, gaia_ID, parallax, pmra, pmra_err, pmdec, pmdec_err, RV, RV_err, #dist, dist_err,
         jr, jr_err,# jphi, jphi_err,
        jz, jz_err, TEFF, TEFF_ERR, LOGG, LOGG_ERR, SNR, ASPCAPFLAG, STARFLAG, FE_H, FE_H_err, C_FE, 
         C_FE_err, CI_FE, CI_FE_err, N_FE, N_FE_err, O_FE, O_FE_err, MG_FE, MG_FE_err, AL_FE, AL_FE_err, SI_FE,
         SI_FE_err, P_FE, P_FE_err, S_FE, S_FE_err, K_FE, K_FE_err, CA_FE, CA_FE_err, TI_FE, TI_FE_err, TIII_FE,
         TIII_FE_err, V_FE, V_FE_err, CR_FE, CR_FE_err, MN_FE, MN_FE_err, CO_FE, CO_FE_err, NI_FE, NI_FE_err]

aspcapflags_filter = np.array(cols[20])==0
starflags_filter = np.array(cols[21])==0

filters = aspcapflags_filter*starflags_filter


filtered_data = []
for c in cols:
    a = np.array(c)[filters]
    filtered_data.append(a)
    
print("There are {} stars in our filtered sample".format(len(filtered_data[0])))

FE_H_filtered, C_FE_filtered, CI_FE_filtered = filtered_data[22], filtered_data[24], filtered_data[26]
N_FE_filtered, O_FE_filtered, MG_FE_filtered = filtered_data[28], filtered_data[30], filtered_data[32]
AL_FE_filtered, SI_FE_filtered, P_FE_filtered = filtered_data[34], filtered_data[36], filtered_data[38]
S_FE_filtered, K_FE_filtered, CA_FE_filtered = filtered_data[40], filtered_data[42], filtered_data[44]
TI_FE_filtered, TIII_FE_filtered, V_FE_filtered = filtered_data[46], filtered_data[48], filtered_data[50]
CR_FE_filtered, MN_FE_filtered, CO_FE_filtered, NI_FE_filtered = filtered_data[52], filtered_data[54], filtered_data[56], filtered_data[58]
RV_filtered  = filtered_data[9]

FE_H_err, C_FE_err, CI_FE_err = filtered_data[23], filtered_data[25], filtered_data[27]
N_FE_err, O_FE_err, MG_FE_err = filtered_data[29], filtered_data[31], filtered_data[33]
AL_FE_err, SI_FE_err, P_FE_err = filtered_data[35], filtered_data[37], filtered_data[39]
S_FE_err, K_FE_err, CA_FE_err = filtered_data[41], filtered_data[43], filtered_data[45]
TI_FE_err, TIII_FE_err, V_FE_err = filtered_data[47], filtered_data[49], filtered_data[51]
CR_FE_err, MN_FE_err, CO_FE_err, NI_FE_err = filtered_data[53], filtered_data[55], filtered_data[57], filtered_data[59]
RV_err  = filtered_data[10]

train_size = round(0.8 * len(filtered_data[0]))
test_size = round(0.1 * len(filtered_data[0]))
validation_size = round(0.1 * len(filtered_data[0]))

training_labels_raw = np.transpose(np.array([FE_H_filtered[:train_size], C_FE_filtered[:train_size], CI_FE_filtered[:train_size], 
                                        N_FE_filtered[:train_size], O_FE_filtered[:train_size], MG_FE_filtered[:train_size],
                                        AL_FE_filtered[:train_size], SI_FE_filtered[:train_size], P_FE_filtered[:train_size],
                                        S_FE_filtered[:train_size], K_FE_filtered[:train_size], CA_FE_filtered[:train_size],
                                        TI_FE_filtered[:train_size], TIII_FE_filtered[:train_size], V_FE_filtered[:train_size], 
                                        CR_FE_filtered[:train_size], MN_FE_filtered[:train_size], CO_FE_filtered[:train_size], NI_FE_filtered[:train_size]]))

error_training_labels_raw = np.transpose(np.array([FE_H_err[:train_size], C_FE_err[:train_size], CI_FE_err[:train_size], 
                                        N_FE_err[:train_size], O_FE_err[:train_size], MG_FE_err[:train_size],
                                        AL_FE_err[:train_size], SI_FE_err[:train_size], P_FE_err[:train_size],
                                        S_FE_err[:train_size], K_FE_err[:train_size], CA_FE_err[:train_size],
                                        TI_FE_err[:train_size], TIII_FE_err[:train_size], V_FE_err[:train_size], 
                                        CR_FE_err[:train_size], MN_FE_err[:train_size], CO_FE_err[:train_size], NI_FE_err[:train_size]]))


test_labels_raw = np.transpose(np.array([FE_H_filtered[train_size: train_size + test_size], C_FE_filtered[train_size: train_size + test_size], 
                                     CI_FE_filtered[train_size: train_size + test_size], N_FE_filtered[train_size: train_size + test_size],
                                     O_FE_filtered[train_size: train_size + test_size], MG_FE_filtered[train_size: train_size + test_size],
                                        AL_FE_filtered[train_size: train_size + test_size], SI_FE_filtered[train_size: train_size + test_size], 
                                     P_FE_filtered[train_size: train_size + test_size], S_FE_filtered[train_size: train_size + test_size], 
                                     K_FE_filtered[train_size: train_size + test_size], CA_FE_filtered[train_size: train_size + test_size],
                                        TI_FE_filtered[train_size: train_size + test_size], TIII_FE_filtered[train_size: train_size + test_size], 
                                     V_FE_filtered[train_size: train_size + test_size], CR_FE_filtered[train_size: train_size + test_size], 
                                     MN_FE_filtered[train_size: train_size + test_size], CO_FE_filtered[train_size: train_size + test_size], 
                                     NI_FE_filtered[train_size: train_size + test_size]]))
                                     
                                     
error_test_labels_raw = np.transpose(np.array([FE_H_err[train_size: train_size + test_size], C_FE_err[train_size: train_size + test_size], CI_FE_err[train_size: train_size + test_size], 
                                        N_FE_err[train_size: train_size + test_size], O_FE_err[train_size: train_size + test_size], MG_FE_err[train_size: train_size + test_size],
                                        AL_FE_err[train_size: train_size + test_size], SI_FE_err[train_size: train_size + test_size], P_FE_err[train_size: train_size + test_size],
                                        S_FE_err[train_size: train_size + test_size], K_FE_err[train_size: train_size + test_size], CA_FE_err[train_size: train_size + test_size],
                                        TI_FE_err[train_size: train_size + test_size], TIII_FE_err[train_size: train_size + test_size], V_FE_err[train_size: train_size + test_size], 
                                        CR_FE_err[train_size: train_size + test_size], MN_FE_err[train_size: train_size + test_size], CO_FE_err[train_size: train_size + test_size], NI_FE_err[train_size: train_size + test_size]]))


validation_labels_raw = np.transpose(np.array([FE_H_filtered[train_size + test_size: train_size + test_size + validation_size], C_FE_filtered[train_size + test_size: train_size + test_size + validation_size], 
                                     CI_FE_filtered[train_size + test_size: train_size + test_size + validation_size], N_FE_filtered[train_size + test_size: train_size + test_size + validation_size],
                                     O_FE_filtered[train_size + test_size: train_size + test_size + validation_size], MG_FE_filtered[train_size + test_size: train_size + test_size + validation_size],
                                        AL_FE_filtered[train_size + test_size: train_size + test_size + validation_size], SI_FE_filtered[train_size + test_size: train_size + test_size + validation_size], 
                                     P_FE_filtered[train_size + test_size: train_size + test_size + validation_size], S_FE_filtered[train_size + test_size: train_size + test_size + validation_size], 
                                     K_FE_filtered[train_size + test_size: train_size + test_size + validation_size], CA_FE_filtered[train_size + test_size: train_size + test_size + validation_size],
                                        TI_FE_filtered[train_size + test_size: train_size + test_size + validation_size], TIII_FE_filtered[train_size + test_size: train_size + test_size + validation_size], 
                                     V_FE_filtered[train_size + test_size: train_size + test_size + validation_size], CR_FE_filtered[train_size + test_size: train_size + test_size + validation_size], 
                                     MN_FE_filtered[train_size + test_size: train_size + test_size + validation_size], CO_FE_filtered[train_size + test_size: train_size + test_size + validation_size], 
                                     NI_FE_filtered[train_size + test_size: train_size + test_size + validation_size]]))
                                     
                                     
error_validation_labels_raw = np.transpose(np.array([FE_H_err[train_size + test_size: train_size + test_size + validation_size], C_FE_err[train_size + test_size: train_size + test_size + validation_size], CI_FE_err[train_size + test_size: train_size + test_size + validation_size], 
                                        N_FE_err[train_size + test_size: train_size + test_size + validation_size], O_FE_err[train_size + test_size: train_size + test_size + validation_size], MG_FE_err[train_size + test_size: train_size + test_size + validation_size],
                                        AL_FE_err[train_size + test_size: train_size + test_size + validation_size], SI_FE_err[train_size + test_size: train_size + test_size + validation_size], P_FE_err[train_size + test_size: train_size + test_size + validation_size],
                                        S_FE_err[train_size + test_size: train_size + test_size + validation_size], K_FE_err[train_size + test_size: train_size + test_size + validation_size], CA_FE_err[train_size + test_size: train_size + test_size + validation_size],
                                        TI_FE_err[train_size + test_size: train_size + test_size + validation_size], TIII_FE_err[train_size + test_size: train_size + test_size + validation_size], V_FE_err[train_size + test_size: train_size + test_size + validation_size], 
                                        CR_FE_err[train_size + test_size: train_size + test_size + validation_size], MN_FE_err[train_size + test_size: train_size + test_size + validation_size], CO_FE_err[train_size + test_size: train_size + test_size + validation_size], NI_FE_err[train_size + test_size: train_size + test_size + validation_size]]))

full_training_labels_raw = np.c_[training_labels_raw, error_training_labels_raw]
full_test_labels_raw = np.c_[test_labels_raw, error_test_labels_raw]
full_validation_labels_raw = np.c_[validation_labels_raw, error_validation_labels_raw]

print(full_training_labels_raw.shape, full_test_labels_raw.shape, full_validation_labels_raw.shape)
print(np.min(full_training_labels_raw.T, axis=1), np.max(full_training_labels_raw.T, axis=1))

# remove errors above thresholds, unreasonably huge rror bars/abundance estimates

err_threshold = 3 # remove errors that are larger than this...
err_mask_train = np.all(np.abs(full_training_labels_raw) < err_threshold, axis=1)
err_mask_test = np.all(np.abs(full_test_labels_raw) < err_threshold, axis=1)
err_mask_validation = np.all(np.abs(full_validation_labels_raw) < err_threshold, axis=1)

full_training_labels = full_training_labels_raw[err_mask_train]
full_test_labels = full_test_labels_raw[err_mask_test]
full_validation_labels = full_validation_labels_raw[err_mask_validation]

print(full_training_labels.shape, full_test_labels.shape, full_validation_labels.shape)
print(np.min(full_training_labels.T, axis=1), np.max(full_training_labels.T, axis=1))

# standardization
from sklearn.preprocessing import StandardScaler

# manual scaler
# standardized_pca_array = []
# standardization_means = np.zeros(len(np.array(x_df_transpose)))
# standardization_sigmas = np.zeros(len(np.array(x_df_transpose)))

# for i, param in enumerate(np.array(x_df_transpose)):
#     mean, sigma = np.mean(param), np.std(param)
#     norm = (param - mean)/sigma
#     standardized_pca_array.append(norm)
#     standardization_means[i] = mean
#     standardization_sigmas[i] = sigma

# standard scaler (same method for t-SNE/UMAP)
# split abundances and errors into separate datasets
training_labels_abundances = full_training_labels[:, :19]
training_labels_errors = full_training_labels[:, 19:]

# split abundances and errors into separate datasets
test_labels_abundances = full_test_labels[:, :19]
test_labels_errors = full_test_labels[:, 19:]

# split abundances and errors into separate datasets
validation_labels_abundances = full_validation_labels[:, :19]
validation_labels_errors = full_validation_labels[:, 19:]

standardized_abundance_training_arr = StandardScaler().fit_transform(training_labels_abundances)
standardized_abundance_test_arr = StandardScaler().fit_transform(test_labels_abundances)
standardized_abundance_validation_arr = StandardScaler().fit_transform(validation_labels_abundances)

#### UMAP:

# Sweep:

# # n neighbors, minimum distances and mses
# # 4 major hyperparameters are neighbors, components, metric, min_dist
# n_neighbors_ls = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
# min_dists = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.99]
# training_mses = np.zeros((len(n_neighbors_ls), len(min_dists)))
# test_mses = np.zeros((len(n_neighbors_ls), len(min_dists)))

# # sweep
# for i, n_neighbors in enumerate(n_neighbors_ls):
#     for j, min_dist in enumerate(min_dists):
#         print(n_neighbors, min_dist)
#         # fit using *training* data
#         # all default except neighbors/distances, unique=True and pca embedding since spectral seems to fail for some of them...
#         reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='euclidean', 
#                             n_epochs=1000, learning_rate=1.0, init='pca', min_dist=min_dist, 
#                             verbose=True, n_jobs=multiprocessing.cpu_count() - 1, unique=True, 
#                             random_state=1234, transform_seed=1234)
#         fit_umap = reducer.fit(standardized_abundance_training_arr)
#         # embed *train* and *test* data
#         embed_train = fit_umap.transform(standardized_abundance_training_arr)
#         embed_test = fit_umap.transform(standardized_abundance_test_arr)
#         # reconstruct *train* and *test* data and record MSE
#         reconstruction_training = fit_umap.inverse_transform(embed_train)
#         reconstruction_test = fit_umap.inverse_transform(embed_test)
#         # record
#         training_mses[i, j] = np.mean((standardized_abundance_training_arr - reconstruction_training) ** 2) * 1000  # MSE multiplied by 1000 to match the VAE ones
#         test_mses[i, j] = np.mean((standardized_abundance_test_arr - reconstruction_test) ** 2) * 1000  # MSE multiplied by 1000 to match the VAE ones

# np.save('umap_training_mses.npy', training_mses)
# np.save('umap_test_mses.npy', test_mses)

# Train a bunch of trials with fixed number seed:

def get_umap_mse(epochs, n_neighbors=100, min_dist=0.0, training_arr=standardized_abundance_training_arr, test_arr=standardized_abundance_test_arr, 
             components=2, metric='euclidean', learning_rate=1.0, initialization='spectral', seed=1234):
    """
    Get the MSE for UMAP given epochs, n_neighbors, min_dist, components, metric, epochs, learning rate, initialization, seed
    and a set of training and test data

    params is tuple of n_neighbors, min_dist
    """
    # fit using *training* data
    n_jobs = multiprocessing.cpu_count() - 1 if seed is None else 1
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=components, metric=metric, 
                        n_epochs=epochs, learning_rate=learning_rate, init=initialization, min_dist=min_dist, 
                        verbose=False, n_jobs=n_jobs, unique=True, 
                        random_state=seed, transform_seed=seed) 
    fit_umap = reducer.fit(training_arr)
    # embed *test* data
    embed_train = fit_umap.transform(training_arr)
    embed_test = fit_umap.transform(test_arr)
    # reconstruct *test* data and record MSE of it
    reconstruction_training = fit_umap.inverse_transform(embed_train)
    reconstruction_test = fit_umap.inverse_transform(embed_test)
    # return MSE
    training_mse = np.mean((standardized_abundance_training_arr - reconstruction_training) ** 2) * 1000  # MSE multiplied by 1000 to match the VAE ones
    test_mse = np.mean((standardized_abundance_test_arr - reconstruction_test) ** 2) * 1000  # MSE multiplied by 1000 to match the VAE ones
    return training_mse, test_mse

#### PERFORM TRAINING:

# epochs
epochs = list(range(100, 5000, 1))
# multiprocess
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
# run
epochs_mses = np.array(list(tqdm(pool.imap(get_umap_mse, epochs), total = len(epochs))))
# join and close
pool.close()
pool.join()
# save
np.save('umap_all_epochs_mses.npy', epochs_mses)
# save training and test mses
training_epochs_mses = epochs_mses[:, 0]
test_epochs_mses = epochs_mses[:, 1]
# save
np.save('umap_training_epochs_full_mses.npy', training_epochs_mses)
np.save('umap_test_epochs_full_mses.npy', test_epochs_mses)

# epochs = range(100, 5000)  # up to 5000 epochs
# n_neighbors=100
# min_dist=0.0
# training_mses_epochs = np.zeros(len(epochs))
# test_mses_epochs = np.zeros(len(epochs))

# for i, epoch in tqdm(enumerate(epochs)):
#     print(epoch)
#     reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='euclidean', 
#                     n_epochs=epoch, learning_rate=1.0, init='spectral', min_dist=min_dist, 
#                     verbose=True, n_jobs=multiprocessing.cpu_count() - 1, unique=True, 
#                     random_state=1234, transform_seed=1234)
#     # fit with training data
#     fit_umap = reducer.fit(standardized_abundance_training_arr)
#     # embed *train* and *test* data
#     embed_train = fit_umap.transform(standardized_abundance_training_arr)
#     embed_test = fit_umap.transform(standardized_abundance_test_arr)
#     # reconstruct *train* and *test* data and record MSE
#     reconstruction_training = fit_umap.inverse_transform(embed_train)
#     reconstruction_test = fit_umap.inverse_transform(embed_test)
#     # record
#     training_mses_epochs[i] = np.mean((standardized_abundance_training_arr - reconstruction_training) ** 2) * 1000  # MSE multiplied by 1000 to match the VAE ones
#     test_mses_epochs[i] = np.mean((standardized_abundance_test_arr - reconstruction_test) ** 2) * 1000  # MSE multiplied by 1000 to match the VAE ones

# np.save('umap_training_mses_epochs.npy', training_mses_epochs)
# np.save('umap_test_mses_epochs.npy', test_mses_epochs)

#### EXTRA:

# def umap_mse(params, training_arr=standardized_abundance_training_arr, test_arr=standardized_abundance_test_arr, 
#              components=2, metric='euclidean', epochs=200, learning_rate=1.0, initialization='pca', seed=1234):
#     """
#     Get the MSE for UMAP given params, components, metric, epochs, learning rate, initialization, seed
#     and a set of training and test data

#     params is tuple of n_neighbors, min_dist
#     """
#     # unpack params
#     n_neighbors, min_dist = params
#     # fit using *training* data
#     reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=components, metric=metric, 
#                         n_epochs=epochs, learning_rate=learning_rate, init=initialization, min_dist=min_dist, 
#                         verbose=True, n_jobs=multiprocessing.cpu_count() - 1, unique=True, 
#                         random_state=seed, transform_seed=seed) 
#     fit_umap = reducer.fit(training_arr)
#     # embed *test* data
#     embedding = fit_umap.transform(test_arr)
#     # reconstruct *test* data and record MSE of it
#     reconstruction = fit_umap.inverse_transform(embedding)
#     # return MSE
#     return np.mean((test_arr - reconstruction) ** 2) * 1000

# n_neighbors, min_dist = 1000, 0.99
# # fit using *training* data
# reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='euclidean', 
#                     n_epochs=200, learning_rate=1.0, init='pca', min_dist=min_dist, 
#                     verbose=True, n_jobs=multiprocessing.cpu_count() - 1, unique=True) 
# fit_umap = reducer.fit(standardized_abundance_training_arr)
# # embed *test* data
# embed_train = fit_umap.transform(standardized_abundance_training_arr)
# embedding = fit_umap.transform(standardized_abundance_test_arr)
# # reconstruct *test* data and record MSE of it
# reconstruction_train = fit_umap.inverse_transform(embed_train)
# reconstruction = fit_umap.inverse_transform(embedding)
# # return MSE
# np.mean((standardized_abundance_training_arr - reconstruction_train) ** 2) * 1000, np.mean((standardized_abundance_test_arr - reconstruction) ** 2) * 1000

# umap_mse((1000, 0.99), epochs=1000)