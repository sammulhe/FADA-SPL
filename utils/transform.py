import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def four_transform(src_data, trg_data, timestep, features, high_per=0.8):
    per = np.random.beta(5, 10)
    #src_data = src_data.view(src_data.shape[0], timestep, features)#.numpy()
    #trg_data = trg_data.view(trg_data.shape[0], timestep, features)#.numpy()

    for feature in range(features):
        src_fft = torch.fft.rfft(src_data[:, feature, :], dim=-1)
        src_amp = torch.abs(src_fft)  ###复数的绝对值（每个频率的振幅）
        src_pha = torch.angle(src_fft)

        trg_fft = torch.fft.rfft(trg_data[:, feature, :], dim=-1)
        trg_amp = torch.abs(trg_fft)  ###复数的绝对值（每个频率的振幅）

        highfreq = int(high_per * (timestep/2 + 1))
        src_trg_amp = src_amp

        for i in range(highfreq, int(timestep/2 + 1)):
            #src_trg_amp[:, i] = src_amp[:, i] * (1 - per) + trg_amp[:, i] * per
            src_trg_amp[:, i] = trg_amp[:, i]


        src_ifft = src_trg_amp * torch.exp(1j * src_pha)  ##ifft是在fft_src_np上逆转换为原图像，但现在分解成振幅和相位，就可以通过这样，还原回fft_src_np,再用ifft逆转换回原图像
        src_ifft = torch.fft.irfft(src_ifft, dim=1).real
        src_data[:, feature, :] = src_ifft

    return src_data



def four_transform_tmp(src_data, trg_data, timestep, features):
    per = np.random.beta(5, 10)
    #src_data = src_data.view(src_data.shape[0], timestep, features)#.numpy()
    #trg_data = trg_data.view(trg_data.shape[0], timestep, features)#.numpy()

    for feature in range(features):
        src_fft = torch.fft.fft(src_data[:, :, feature], dim=1)
        src_amp = torch.abs(src_fft)  ###复数的绝对值（每个频率的振幅）
        src_pha = torch.angle(src_fft)

        trg_fft = torch.fft.fft(trg_data[:, :, feature], dim=1)
        trg_amp = torch.abs(trg_fft)  ###复数的绝对值（每个频率的振幅）

        ####把高频率的src和trg替换####################################
        #src_mean = np.percentile(src_amp, 0, axis=1).reshape(-1, 1)  # np.mean(a, axis=1).reshape(-1, 1)
        # src_mean = np.repeat(src_mean, timestep, axis=1)
        src_mean = torch.quantile(src_amp, per, dim=1, keepdim=True)
        #print(src_mean)
        if len(src_data) == len(trg_data):
            src_trg_amp = torch.where(src_amp >= src_mean, src_amp, trg_amp)
            # src_trg_amp = src_amp * src_indices + trg_amp * trg_indices
        else:
            src_indices = torch.where(src_amp >= src_mean, True, False)
            trg_indices = ~src_indices

            src_trg_amp = torch.empty(src_amp.shape, dtype=src_amp.dtype)
            j = 0
            for i in range(len(src_amp)):
                src_trg_amp[i, :] = src_amp[i, :] * src_indices[i, :] + trg_amp[j, :] * trg_indices[i, :]
                j += 1
                if j == len(trg_amp):
                    j = 0

        src_ifft = src_trg_amp * torch.exp(1j * src_pha)  ##ifft是在fft_src_np上逆转换为原图像，但现在分解成振幅和相位，就可以通过这样，还原回fft_src_np,再用ifft逆转换回原图像
        src_ifft = torch.fft.ifft(src_ifft, dim=1).real
        src_data[:, :, feature] = src_ifft

    return src_data

def four_transform2(src_data, trg_data, timestep, features):
    per = np.random.normal(0, 1)

    for feature in range(features):
        src_fft = torch.fft.fft(src_data[:, :, feature], dim=1)
        src_amp = torch.abs(src_fft)  ###复数的绝对值（每个频率的振幅）
        src_pha = torch.angle(src_fft)

        trg_fft = torch.fft.fft(trg_data[:, :, feature], dim=1)
        trg_amp = torch.abs(trg_fft)  ###复数的绝对值（每个频率的振幅）

        if len(src_data) == len(trg_data):
            src_trg_amp = (1 - per) * src_amp + per * trg_amp
        else:
            src_trg_amp = torch.empty(src_amp.shape, dtype=src_amp.dtype)
            j = 0
            for i in range(len(src_amp)):
                #src_trg_amp[i, :] = src_amp[i, :] * src_indices[i, :] + trg_amp[j, :] * trg_indices[i, :]
                src_trg_amp[i, :] = src_amp[i, :] * (1 - per) + trg_amp[j, :] * per
                j += 1
                if j == len(trg_amp):
                    j = 0

        src_ifft = src_trg_amp * torch.exp(1j * src_pha)  ##ifft是在fft_src_np上逆转换为原图像，但现在分解成振幅和相位，就可以通过这样，还原回fft_src_np,再用ifft逆转换回原图像
        src_ifft = torch.fft.ifft(src_ifft, dim=1).real

        src_data[:, :, feature] = src_ifft

    return src_data


def four_transform3(src_data, trg_data, timestep, features, per=0.05):
    src_data = src_data.view(src_data.shape[0], timestep, features)#.numpy()
    trg_data = trg_data.view(trg_data.shape[0], timestep, features)#.numpy()

    for feature in range(features):
        x = torch.fft.fftfreq(timestep, 1 / 50)

        src_fft = torch.fft.fft(src_data[:, :, feature], dim=1)
        src_amp = torch.abs(src_fft)  ###复数的绝对值（每个频率的振幅）
        src_pha = torch.angle(src_fft)

        trg_fft = torch.fft.fft(trg_data[:, :, feature], dim=1)
        trg_amp = torch.abs(trg_fft)  ###复数的绝对值（每个频率的振幅）
        # trg_pha = np.angle(trg_fft)
        # print(src_mean)

        src_indices = np.abs(x) >= int(per * len(x))  # abs_y[i, :] >= 10
        trg_indices = np.abs(x) <= int(per* len(x))
        if len(src_data) == len(trg_data):
            src_trg_amp = src_amp * src_indices + trg_amp * trg_indices
            ##src_trg_amp = (1 - per) * src_amp + per * trg_amp
        else:
            src_trg_amp = torch.empty(src_amp.shape, dtype=src_amp.dtype)
            j = 0
            for i in range(len(src_amp)):
                src_trg_amp[i, :] = src_amp[i, :] * src_indices + trg_amp[j, :] * trg_indices
                j += 1
                if j == len(trg_amp):
                    j = 0
        src_ifft = src_trg_amp * torch.exp(
            1j * src_pha)  ##ifft是在fft_src_np上逆转换为原图像，但现在分解成振幅和相位，就可以通过这样，还原回fft_src_np,再用ifft逆转换回原图像
        src_ifft = torch.fft.ifft(src_ifft, dim=1).real

        # print(src_ifft.shape)
        src_data[:, :, feature] = src_ifft

    src_data = src_data.view(src_data.shape[0], timestep * features)
    #src_data = torch.from_numpy(src_data)

    #print('bb', src_data)
    return src_data





def test_fourier(src_data, timestep, features, per=0.05):
    #print('aaa', src_data)
    src_data = src_data.view(src_data.shape[0], timestep, features)#.numpy()
    #trg_data = trg_data.view(trg_data.shape[0], timestep, features)#.numpy()

    for feature in range(features):
        #x = torch.fft.fftfreq(timestep, 1 / 50)
        # x = np.arange(timestep)

        src_fft = torch.fft.fft(src_data[:, :, feature], dim=1)
        src_amp = torch.abs(src_fft)  ###复数的绝对值（每个频率的振幅）
        src_pha = torch.angle(src_fft)


        src_ifft = 1 * np.exp(1j * src_pha)  ##ifft是在fft_src_np上逆转换为原图像，但现在分解成振幅和相位，就可以通过这样，还原回fft_src_np,再用ifft逆转换回原图像
        src_ifft = torch.fft.ifft(src_ifft, dim=1).real

        # print(src_ifft.shape)
        src_data[:, :, feature] = src_ifft

    src_data = src_data.view(src_data.shape[0], timestep * features)
    #src_data = torch.from_numpy(src_data)

    #print('bb', src_data)
    return src_data


