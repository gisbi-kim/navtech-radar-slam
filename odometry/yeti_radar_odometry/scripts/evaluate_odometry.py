import numpy as np
import csv
import sys
import os
import matplotlib.pyplot as plt
from plot_scan_matching_accuracy import *

lengths = [100, 200, 300, 400, 500, 600, 700, 800]

# Calculates path length along the trajectory
def trajectoryDistances(poses):
    dist = [0]
    for i in range(1, len(poses)):
        P1 = poses[i - 1]
        P2 = poses[i]
        dx = P1[0, 2] - P2[0, 2]
        dy = P1[1, 2] - P2[1, 2]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1

def rotationError(pose_error):
    return abs(np.arcsin(pose_error[0, 1]))

def translationError(pose_error):
    return np.sqrt(pose_error[0, 2]**2 + pose_error[1, 2]**2)

def get_inverse_tf(T):
    T2 = np.identity(3)
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    t = np.reshape(t, (2, 1))
    T2[0:2, 0:2] = R.transpose()
    t = np.matmul(-1 * R.transpose(), t)
    T2[0, 2] = t[0]
    T2[1, 2] = t[1]
    return T2

def calcSequenceErrors(poses_gt, poses_res):
    err = []
    step_size = 4 # Every second
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)
    print(dist[-1])

    for first_frame in range(0, len(poses_gt), step_size):
        for i in range(0, len(lengths)):
            length = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(get_inverse_tf(poses_gt[first_frame]), poses_gt[last_frame])
            pose_delta_res = np.matmul(get_inverse_tf(poses_res[first_frame]), poses_res[last_frame])
            pose_error = np.matmul(get_inverse_tf(pose_delta_res), pose_delta_gt)
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err

def calcAbsoluteTrajectoryError(poses_gt, poses_res):
    error = 0
    for T_gt, T_res in zip(poses_gt, poses_res):
        T_err = np.matmul(get_inverse_tf(T_res), T_gt)
        t_err = T_err[0:2, 2].reshape(2, 1)
        error += (np.linalg.norm(t_err) ** 2)
    error /= len(poses_gt)
    return np.sqrt(error)

def saveSequenceErrors(err, file_name):
    with open(file_name, "w") as f:
        for e in err:
            f.write("{},{},{},{},{}\n".format(e[0], e[1], e[2], e[3], e[4]))

def loadSequenceErrors(file_name):
    err = []
    with open(file_name, "r") as f:
        for line in f:
            e = []
            parts = line.split(',')
            for part in parts:
                e.append(float(part))
            err.append(e)
    return err

def saveErrorPlots(errlist, filename):
    matplotlib.rcParams.update({'font.size': 13, 'xtick.labelsize' : 12, 'ytick.labelsize' : 12,
        'axes.linewidth' : 1.5, 'font.family' : 'serif', 'pdf.fonttype' : 42})
    fig, axs = plt.subplots(2, 2, tight_layout=True)
    axs[0, 0].set_title('T. Err (%) vs. Length (m)')
    axs[0, 0].grid(which='both', linestyle='--', alpha=0.5)
    axs[0, 1].set_title('R. Err (deg/m) vs. Length (m)')
    axs[0, 1].grid(which='both', linestyle='--', alpha=0.5)
    axs[1, 0].set_title('T. Err (%) vs. Speed (m/s)')
    axs[1, 0].grid(which='both', linestyle='--', alpha=0.5)
    axs[1, 1].set_title('R. Err (deg/m) vs. Speed (m/s)')
    axs[1, 1].grid(which='both', linestyle='--', alpha=0.5)

    for j in range(0, len(errlist)):
        err = errlist[j]
        t_len_err = []
        r_len_err = []
        t_vel_err = []
        r_vel_err = []
        for i in range(0, len(lengths)):
            length = lengths[i]
            num = 0
            t_err = 0
            r_err = 0
            for e in err:
                if e[3] - length < 1.0:
                    t_err += e[2]
                    r_err += e[1]
                    num += 1
            if num == 0:
                break
            t_len_err.append(100 * t_err / float(num))
            r_len_err.append(180 * r_err / (float(num) * np.pi))

        for v in range(0, 26):
            num = 0
            t_err = 0
            r_err = 0
            for e in err:
                if v - e[4] < 2.0:
                    t_err += e[2]
                    r_err += e[1]
                    num += 1
            if num == 0 or num < 500:
                break
            t_vel_err.append(100 * t_err / float(num))
            r_vel_err.append(180 * r_err / (float(num) * np.pi))

        vx = np.arange(0, 26, 1)
        l = len(t_len_err)
        m = len(t_vel_err)
        if j == 0:
            axs[0, 0].plot(lengths[:l], t_len_err, 'ob-', label='RIGID', linewidth=2.0)
            axs[0, 1].plot(lengths[:l], r_len_err, 'ob-', label='RIGID', linewidth=2.0)
            axs[1, 0].plot(vx[:m], t_vel_err, 'ob-', label='RIGID', linewidth=2.0)
            axs[1, 1].plot(vx[:m], r_vel_err, 'ob-', label='RIGID', linewidth=2.0)
        if j == 1:
            axs[0, 0].plot(lengths[:l], t_len_err, 'Dr-', label='MC-RANSAC', linewidth=2.0)
            axs[0, 1].plot(lengths[:l], r_len_err, 'Dr-', label='MC-RANSAC', linewidth=2.0)
            axs[1, 0].plot(vx[:m], t_vel_err, 'Dr-', label='MC-RANSAC', linewidth=2.0)
            axs[1, 1].plot(vx[:m], r_vel_err, 'Dr-', label='MC-RANSAC', linewidth=2.0)
        if j == 2:
            axs[0, 0].plot(lengths[:l], t_len_err, 'og-', label='MC+DOPP', linewidth=2.0)
            axs[0, 1].plot(lengths[:l], r_len_err, 'og-', label='MC+DOPP', linewidth=2.0)
            axs[1, 0].plot(vx[:m], t_vel_err, 'og-', label='MC+DOPP', linewidth=2.0)
            axs[1, 1].plot(vx[:m], r_vel_err, 'og-', label='MC+DOPP', linewidth=2.0)

    axLine, axLabel = axs[0, 0].get_legend_handles_labels()

    fig.legend(axLine, axLabel, loc = 'upper right', bbox_to_anchor=(0.925, 0.91), fontsize='small')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)


def getStats(err):
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err

if __name__ == '__main__':

    dontinclude = ['accuracy2019-01-10-14-36-48-radar-oxford-10k-partial',
        'accuracy2019-01-10-11-46-21-radar-oxford-10k',
        'accuracy2019-01-16-14-15-33-radar-oxford-10k']

    folder = './icra_odom/'
    ff = os.listdir('./icra_odom/')
    files = []
    for f in ff:
        if 'accuracy' in f:
            bad = False
            for dont in dontinclude:
                if dont in f:
                    bad = True
            if not bad:
                files.append(folder + f)

    # files = ['accuracy.csv']

    err_rigid = []
    err_md = []
    err_dopp = []

    ate_rigid = []
    ate_md = []
    ate_dopp = []

    for file in files:
	    print(file)
        T_gt = np.identity(3)
        T_res = np.identity(3)
        T_md = np.identity(3)
        T_dopp = np.identity(3)
        poses_gt = []
        poses_res = []
        poses_md = []
        poses_dopp = []
        with open(file) as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            for row in reader:
                if i == 0:
                    i = 1
                    continue
                # Create transformation matrices
                T_gt_ = get_transform(float(row[3]), float(row[4]), float(row[5]))
                T_res_ = get_transform(float(row[0]), float(row[1]), float(row[2]))
                T_md_ = get_transform(float(row[8]), float(row[9]), float(row[10]))
                T_dopp_ = get_transform(float(row[11]), float(row[12]), float(row[13]))
                T_gt = np.matmul(T_gt, T_gt_)
                T_res = np.matmul(T_res, T_res_)
                T_md = np.matmul(T_md, T_md_)
                T_dopp = np.matmul(T_dopp, T_dopp_)

                R_gt = T_gt[0:2,0:2]
                R_res = T_res[0:2,0:2]
                R_md = T_md[0:2,0:2]
                R_dopp = T_dopp[0:2,0:2]
                if np.linalg.det(R_gt) != 1.0:
                    enforce_orthogonality(R_gt)
                    T_gt[0:2,0:2] = R_gt
                if np.linalg.det(R_res) != 1.0:
                    enforce_orthogonality(R_res)
                    T_res[0:2,0:2] = R_res
                if np.linalg.det(R_md) != 1.0:
                    enforce_orthogonality(R_md)
                    T_md[0:2,0:2] = R_md
                if np.linalg.det(R_dopp) != 1.0:
                    enforce_orthogonality(R_dopp)
                    T_dopp[0:2,0:2] = R_dopp

                poses_gt.append(T_gt)
                poses_res.append(T_res)
                poses_md.append(T_md)
                poses_dopp.append(T_dopp)

        err_rigid.extend(calcSequenceErrors(poses_gt, poses_res))
        err_md.extend(calcSequenceErrors(poses_gt, poses_md))
        err_dopp.extend(calcSequenceErrors(poses_gt, poses_dopp))

        ate_rigid.append(calcAbsoluteTrajectoryError(poses_gt, poses_res))
        ate_md.append(calcAbsoluteTrajectoryError(poses_gt, poses_md))
        ate_dopp.append(calcAbsoluteTrajectoryError(poses_gt, poses_dopp))

    saveSequenceErrors(err_rigid, 'pose_error_rigid_icra3.csv')
    saveSequenceErrors(err_md, 'pose_error_mdransac_icra3.csv')
    saveSequenceErrors(err_dopp, 'pose_error_dopp_icra3.csv')
    # err_rigid = loadSequenceErrors('pose_error_rigid_icra2.csv')
    # err_md = loadSequenceErrors('pose_error_mdransac_icra2.csv')
    # err_dopp = loadSequenceErrors('pose_error_dopp_icra2.csv')
    saveErrorPlots([err_rigid, err_md, err_dopp], 'pose_error.pdf')

    ate_rigid = np.array(ate_rigid)
    ate_md = np.array(ate_md)
    ate_dopp = np.array(ate_dopp)

    t_err, r_err = getStats(err_rigid)
    print('RIGID:')
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))
    print('ATE: {} m'.format(np.mean(ate_rigid)))
    t_err, r_err = getStats(err_md)
    print('MC-RANSAC:')
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))
    print('ATE: {} m'.format(np.mean(ate_md)))
    t_err, r_err = getStats(err_dopp)
    print('DOPPLER:')
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))
    print('ATE: {} m'.format(np.mean(ate_dopp)))
