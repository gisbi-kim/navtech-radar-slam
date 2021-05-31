import os
import argparse
import numpy as np

def roll(r):
    return np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]], dtype=np.float64)

def pitch(p):
    return np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]], dtype=np.float64)

def yaw(y):
    return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)

def yawPitchRollToRot(y, p, r):
    Y = yaw(y)
    P = pitch(p)
    R = roll(r)
    C = np.matmul(P, Y)
    return np.matmul(R, C)

def rotToYawPitchRoll(C, eps = 1e-15):
    i = 2
    j = 1
    k = 0
    c_y = np.sqrt(C[i, i]**2 + C[j, i]**2)
    if c_y > eps:
        r = np.arctan2(C[j, i], C[i, i])
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(C[k, j], C[k, k])
    else:
        r = 0
        p = np.arctan2(-C[k, i], c_y)
        y = np.arctan2(-C[j, k], C[j, j])
    return y, p, r

def get_transform(gt):
    T = np.identity(4, dtype=np.float64)
    C_enu_sensor = yawPitchRollToRot(gt[10], gt[9], gt[8])
    T[0, 3] = gt[2]
    T[1, 3] = gt[3]
    T[2, 3] = gt[4]
    T[0:3, 0:3] = C_enu_sensor
    return T

def get_inverse_tf(T):
    T2 = np.identity(4, dtype=np.float64)
    C = T[0:3, 0:3]
    T2[0:3, 0:3] = C.transpose()
    T2[0:3, 3] = (-1 * np.matmul(C.transpose(), T[0:3, 3].reshape(3, 1))).squeeze()
    return T2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/media/backup2/2020_11_26/', type=str,
                        help='path to boreas localization dataset')
    args = parser.parse_args()
    radar_files = os.listdir(args.root + 'radar/')
    radar_files.sort()

    # ROS times that correspond to the forward and reverse traversals respectively
    valid_times = [[1606417372, 1606417597], [1606417663, 1606417877]]  # 2020-11-26

    # ROSTime,GPSTime,x,y,z,vel_x,vel_y,vel_z,roll,pitch,heading,ang_vel_z
    with open(args.root + 'applanix/radar_poses.csv', 'r') as f:
        f.readline()  # Clear out the csv header
        gtlines = f.readlines()

    groundtruth1 = []  # forward
    groundtruth2 = []  # reverse

    def parse(gps_line):
        out = [float(x) for x in gps_line.split(',')]
        out[0] = int(gps_line.split(',')[0])
        return out

    for line in gtlines:
        gt = parse(line)
        timestamp = gt[0] / 1.0e9
        if valid_times[0][0] <= timestamp and timestamp <= valid_times[0][1]:
            groundtruth1.append(gt)
        elif valid_times[1][0] <= timestamp and timestamp <= valid_times[1][1]:
            groundtruth2.append(gt)

    outfile = args.root + 'applanix/radar_loc_gt.csv'
    f = open(outfile, 'w')
    f.write('time1,time2,x,y,yaw,vx1,vy1,w1,vx2,vy2,w2\n')

    minvel = 10.0

    for gt1 in groundtruth1:
        v = np.sqrt(gt1[5]**2 + gt1[6]**2)
        if v < minvel:
            continue
        x1 = gt1[2]
        y1 = gt1[3]

        mind = 6.5
        closest = -1
        for i, gt2 in enumerate(groundtruth2):
            v2 = np.sqrt(gt2[5]**2 + gt2[6]**2)
            if v2 < minvel:
                continue
            x2 = gt2[2]
            y2 = gt2[3]
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if d < mind:
                mind = d
                closest = i

        if closest >= 0:
            T_enu_r1 = get_transform(gt1)
            gt2 = groundtruth2[closest]
            T_enu_r2 = get_transform(gt2)

            # Note: ground truth is 3D, we convert this to 2D for radar data

            T_r1_r2 = np.matmul(get_inverse_tf(T_enu_r1), T_enu_r2)
            r_2_1_in_1 = T_r1_r2[:3, 3]
            x = r_2_1_in_1[0]
            y = r_2_1_in_1[1]
            heading, _, _ = rotToYawPitchRoll(T_r1_r2[0:3, 0:3])

            # get velocity vector in radar sensor frames
            vbar1_enu = np.array([gt1[5], gt1[6], gt1[7]]).reshape(3, 1)
            C_enu_r1 = T_enu_r1[0:3, 0:3]
            vbar1_r1 = np.matmul(C_enu_r1.transpose(), vbar1_enu)
            vx_r1 = vbar1_r1[0, 0]
            vy_r1 = vbar1_r1[1, 0]

            vbar2_enu = np.array([gt2[5], gt2[6], gt2[7]]).reshape(3, 1)
            C_enu_r2 = T_enu_r2[0:3, 0:3]
            vbar2_r2 = np.matmul(C_enu_r2.transpose(), vbar2_enu)
            vx_r2 = vbar2_r2[0, 0]
            vy_r2 = vbar2_r2[1, 0]

            # Angular velocity should already be in the radar frame
            # Note: z-axis for radar points down!
            w_r1 = gt1[11]
            w_r2 = gt2[11]

            f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(gt1[0], gt2[0], x, y, heading, vx_r1, vy_r1, w_r1,
                                                                vx_r2, vy_r2, w_r2))

    f.close()
