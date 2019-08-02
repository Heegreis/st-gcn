import sys
sys.path.append('/data/paperProjects/Heegreis/st-gcn')
from feeder.feeder import Feeder

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from tensorboardX import SummaryWriter

def showSkeleton(data):
    data = data.permute(0, 2, 1, 3) # n,c,t,v -> n, t, c, v
    all_action = data.numpy() # n, t, c, v

    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                        (22, 23), (23, 8), (24, 25), (25, 12)]

    for action in all_action:   # t, c, v
        for skeleton in action: # c, v
            # get one fig data(every point)
            ## set x, z be ground: 
            X = skeleton[0]
            Z = skeleton[1]
            Y = skeleton[2]

            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter(X, Y, Z)

            for neighbor in neighbor_1base:
                index1 = neighbor[0] - 1
                index2 = neighbor[1] - 1
                
                x1 = X[index1]
                y1 = Y[index1]
                z1 = Z[index1]

                x2 = X[index2]
                y2 = Y[index2]
                z2 = Z[index2]

                ax.plot3D([x1, x2], [y1, y2], [z1, z2], 'b')

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            plt.title('3D Scatter', color='b', fontsize=20)
            
            # fit the scale
            ax.set_aspect('equal')
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.show()

def showSkeleton2TensorboardX(data, label, writer):
    data = data.permute(0, 2, 1, 3) # n,c,t,v -> n, t, c, v
    all_action = data.numpy() # n, t, c, v

    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                        (22, 23), (23, 8), (24, 25), (25, 12)]

    for action in all_action:   # t, c, v
        imgs = torch.zeros(0, 3, 800, 800)
        for skeleton in action: # c, v
            # get one fig data(every point)
            ## set x, z be ground: 
            X = skeleton[0]
            Z = skeleton[1]
            Y = skeleton[2]

            fig = plt.figure(figsize=(10, 10))
            canvas = FigureCanvas(fig)
            ax = Axes3D(fig)

            ax.scatter(X, Y, Z)

            for neighbor in neighbor_1base:
                index1 = neighbor[0] - 1
                index2 = neighbor[1] - 1
                
                x1 = X[index1]
                y1 = Y[index1]
                z1 = Z[index1]

                x2 = X[index2]
                y2 = Y[index2]
                z2 = Z[index2]

                ax.plot3D([x1, x2], [y1, y2], [z1, z2], 'b')

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            plt.title('3D Scatter', color='b', fontsize=20)
            
            # fit the scale
            ax.set_aspect('equal')
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            width, height = fig.get_size_inches() * fig.get_dpi()
            canvas.draw()
            canvasData = canvas.tostring_rgb()
            image = np.fromstring(canvasData, dtype='uint8').reshape(int(height), int(width), 3)

            img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor)
            # plt.show()
            imgs = torch.cat([imgs, img], dim=0)
            
            # writer.add_figure('matplotlib', fig)
        imgs = imgs.unsqueeze(0)
        writer.add_video('action visualization', imgs, 0)

if __name__ == "__main__":
    data_loader = dict()

    data_path = './data/NTU-RGB-D/xsub/train_data.npy'
    label_path = './data/NTU-RGB-D/xsub/train_label.pkl'

    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=True,
        drop_last=True)
    
    loader = data_loader['train']

    data_bn = nn.BatchNorm1d(3 * 25)

    writer = SummaryWriter()
    i = 0
    for data, label in loader:
        i = i + 1
        x = data

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        showSkeletonTotbx(x, label, writer)
        break
        print(i)
    
    writer.close()