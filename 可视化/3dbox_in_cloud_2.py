import numpy as np
import seaborn as sns
import mayavi.mlab as mlab
import os

colors = sns.color_palette('Paired', 9 * 2)
colors2 = sns.color_palette('Set2', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian',
         'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']


def draw_box(line, in_color=colors):
    line = line.split()
    if(len(line) is 15):  # lable file
        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
    else:  # eval file
        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, conf = line
    h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
    if lab != 'DontCare':
        x_corners = [l / 2, l / 2, -l / 2, -
                     l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2,
                     w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # transform the 3d bbox from object coordiante to camera_0 coordinate
        R = np.array([[np.cos(rot), 0, np.sin(rot)],
                      [0, 1, 0],
                      [-np.sin(rot), 0, np.cos(rot)]])
        corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

        # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
        corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

        def draw(p1, p2, front=1):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=in_color[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)

        # draw the upper 4 horizontal lines
        draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
        draw(corners_3d[1], corners_3d[2])
        draw(corners_3d[2], corners_3d[3])
        draw(corners_3d[3], corners_3d[0])

        # draw the lower 4 horizontal lines
        draw(corners_3d[4], corners_3d[5], 0)
        draw(corners_3d[5], corners_3d[6])
        draw(corners_3d[6], corners_3d[7])
        draw(corners_3d[7], corners_3d[4])

        # draw the 4 vertical lines
        draw(corners_3d[4], corners_3d[0], 0)
        draw(corners_3d[5], corners_3d[1], 0)
        draw(corners_3d[6], corners_3d[2])
        draw(corners_3d[7], corners_3d[3])


if __name__ == '__main__':

    # init
    mlab.options.offscreen = True
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))

    for i in range(100):

        file_id = "%06d" % (i)

        # load point clouds
        # scan_dir = f'data/kitti/object/training/velodyne_reduced/{file_id}.bin'
        scan_dir = f'velodyne_reduced/{file_id}.bin'
        if not os.path.exists(scan_dir):
          print("skip %s for pc" %file_id)
          continue
        scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
        # scan_dir2 = f'data/kitti/object/training_pseudo/velodyne_reduced/{file_id}.bin'
        # scan2 = np.fromfile(scan_dir2, dtype=np.float32).reshape(-1, 4)

        # load evals
        # eval_dir = f'data/test_label2/{file_id}.txt'
        eval_dir = f'comparison1_output_data/{file_id}.txt'
        if not os.path.exists(eval_dir):
          print("skip %s for evals" %file_id)
          continue
        with open(eval_dir, 'r') as f:
          evals = f.readlines()

        # load labels
        # label_dir = f'data/kitti/object/training/label_2/{file_id}.txt'
        label_dir = f'label_2_reduced/{file_id}.txt'
        with open(label_dir, 'r') as f:
          labels = f.readlines()

        # draw point cloud
        plot = mlab.points3d(
            scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig, color=(1, 0, 0))
        # plot = mlab.points3d(
        #     scan2[:, 0], scan2[:, 1], scan2[:, 2], mode="point", figure=fig, color=(0, 1, 0))

        for line in labels:
            draw_box(line, colors)
        mlab.view(azimuth=230, distance=100)
        mlab.savefig(filename='compare1/GT/%s_GT.png' % file_id)
        print("Saved %s.png" % file_id)

        for line in evals:
          draw_box(line,colors)
        mlab.view(azimuth=230, distance=100)
        mlab.savefig(filename='compare1/PD/%s_Eval.png' % file_id)
        print("Saved %s_Eval.png" % file_id)

        mlab.clf(fig)

    mlab.close(all=True)
