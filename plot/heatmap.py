import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import zoom

# heap map size 46, 55, 46, MRI size 181, 217, 181
# thus increase the size of heatmap by 4 times to show in the ImageGrid with the same scale

def resize(mri):
    x, y = mri.shape
    return zoom(mri, (181.0*181.0/(217.0*x), 181.0/y))

def upsample(heat):
    new_heat = np.zeros((46*4, 55*4, 46*4))
    for start_idx1 in range(4):
        for start_idx2 in range(4):
            for start_idx3 in range(4):
                new_heat[start_idx1::4, start_idx2::4, start_idx3::4] = heat
    return new_heat[:181, :217, :181]

def plot_heatmap(path, title, exp_idx, figsize):
    heat_train = upsample(np.load(path + '{}/'.format(exp_idx) + 'train_MCC.npy'))
    heat_valid = upsample(np.load(path + '{}/'.format(exp_idx) + 'valid_MCC.npy'))
    heat_test = upsample(np.load(path + '{}/'.format(exp_idx) + 'test_MCC.npy'))
    heat_NACC = upsample(np.load(path + '{}/'.format(exp_idx) + 'NACC_MCC.npy'))
    heat_AIBL = upsample(np.load(path + '{}/'.format(exp_idx) + 'AIBL_MCC.npy'))
    MRI = np.load('/data/datasets/ADNI_NoBack/ADNI_128_S_1409_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070821114304781_S33787_I69400.npy')

    fig = plt.figure(figsize=figsize, dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3,6),
                     axes_pad=0.00,
                     aspect = True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )
    # Add data to image grid
    small = 0.1

    font_dict = {'fontweight': 'bold', 'fontsize': 14}
    titlename = ['Train', 'Valid', 'Test', 'AIBL', 'NACC']

    im = grid[0].imshow(MRI[:, :, 40].transpose((1, 0))[::-1, :], cmap = 'gray', vmin=-1, vmax=2.5)
    grid[0].axis('off')
    grid[0].set_title("   MRI    ", fontdict=font_dict, loc='right', color = "k")

    for idx, heatmap in enumerate([heat_train, heat_valid, heat_test, heat_AIBL, heat_NACC]):
        im = grid[1+idx].imshow(heatmap[:, :, 40].transpose((1, 0))[::-1, :], cmap = 'hot', vmin=small, vmax=1.0)
        grid[1+idx].axis('off')
        grid[1+idx].set_title(titlename[idx], fontdict=font_dict, loc='center', color = "k")

    im = grid[6].imshow(np.rot90(MRI[:, 100, :]), cmap = 'gray', vmin=-1, vmax=2.5)
    grid[6].axis('off')
    for idx, heatmap in enumerate([heat_train, heat_valid, heat_test, heat_AIBL, heat_NACC]):
        im = grid[7+idx].imshow(np.rot90(heatmap[:, 100, :]), cmap = 'hot', vmin=small, vmax=1.0)
        grid[7+idx].axis('off')

    im = grid[12].imshow(resize(np.rot90(MRI[48, :, :])), cmap = 'gray', vmin=-1, vmax=2.5)
    grid[12].axis('off')
    for idx, heatmap in enumerate([heat_train, heat_valid, heat_test, heat_AIBL, heat_NACC]):
        im = grid[13+idx].imshow(resize(np.rot90(heatmap[48, :, :])), cmap = 'hot', vmin=small, vmax=1.0)
        grid[13+idx].axis('off')

    cbar = grid[7].cax.colorbar(im, drawedges=False)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(14)

    fig.savefig('./{}{}.png'.format(title, exp_idx), dpi=50)

if __name__ == '__main__':
    plot_heatmap('../DPMs/fcn_Aug_exp',title='fcn_Aug', exp_idx=0, figsize=(9, 4))
    # plot_complete_heatmap('../DPMs/fcn_exp', exp_idx=0, figsize=(3, 2))
    # plot_heatmap('../DPMs/fcn_exp', exp_idx=1, figsize=(9, 4))
    # plot_complete_heatmap('../DPMs/fcn_exp', exp_idx=1, figsize=(3, 2))
    # plot_heatmap('../DPMs/fcn_exp', exp_idx=2, figsize=(9, 4))
    # plot_complete_heatmap('../DPMs/fcn_exp', exp_idx=2, figsize=(3, 2))
    # plot_heatmap('../DPMs/fcn_exp', exp_idx=3, figsize=(9, 4))
    # plot_complete_heatmap('../DPMs/fcn_exp', exp_idx=3, figsize=(3, 2))
    # plot_heatmap('../DPMs/fcn_exp', exp_idx=4, figsize=(9, 4))
    # plot_complete_heatmap('../DPMs/fcn_exp', exp_idx=4, figsize=(3, 2))