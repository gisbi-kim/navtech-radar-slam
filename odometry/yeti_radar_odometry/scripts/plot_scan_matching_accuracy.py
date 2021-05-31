import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def enforce_orthogonality(R):
    epsilon = 0.001
    if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
        print("ERROR: this is not a proper rigid transformation!")
    a = (R[0, 0] + R[1, 1]) / 2;
    b = (-R[1, 0] + R[0, 1]) / 2;
    sum = np.sqrt(a**2 + b**2);
    a /= sum;
    b /= sum;
    R[0, 0] = a; R[0, 1] = b;
    R[1, 0] = -b; R[1, 1] = a;

# def get_transform(x, y, theta):
#     R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     if np.linalg.det(R) != 1.0:
#         enforce_orthogonality(R)
#     xbar = np.array([x, y])
#     xbar = np.reshape(xbar, (2, 1))
#     T = np.identity(3)
#     T[0:2, 0:2] = R
#     xbar = np.matmul(-R, xbar)
#     T[0, 2] = xbar[0]
#     T[1, 2] = xbar[1]
#     return T

def get_transform(x, y, theta):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    if np.linalg.det(R) != 1.0:
        enforce_orthogonality(R)
    T = np.identity(3)
    T[0:2, 0:2] = R
    T[0, 2] = x
    T[1, 2] = y
    return T

def get_ins_transformation(time1, time2, gpsfile):

    assert(time2 > time1)
    time1vars = []
    time2vars = []

    def get_vars(row, x1, y1, t1, theta1, time):
        t2 = int(row[0])
        x2 = float(row[5])
        y2 = float(row[6])
        theta2 = float(row[14])
        delta = float(time1 - t1) / float(t2 - t1)
        x = x1 + (x2 - x1) * delta
        y = y1 + (y2 - y1) * delta
        theta = theta1 + (theta2 - theta1) * delta
        return [x, y, theta]

    with open(gpsfile) as f:
        reader = csv.reader(f, delimiter=',')
        # Find timestamps in GPS file that bound time1
        x1 = 0
        y1 = 0
        t1 = 0
        theta1 = 0
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            if (int(row[0]) - time1) > 0 and len(time1vars) == 0:
                time1vars = [x1, y1, theta1]
                # time1vars = get_vars(row, x1, y1, t1, theta1, time1)
            if (int(row[0]) - time2) > 0 and len(time2vars) == 0:
                # time2vars = get_vars(row, x1, y1, t1, theta1, time2)
                time2vars = [x1, y1, theta1]
                break

            t1 = int(row[0])
            x1 = float(row[5])
            y1 = float(row[6])
            theta1 = float(row[14])

    xbar = np.array([time2vars[0] - time1vars[0], time2vars[1] - time1vars[1]])
    xbar = np.reshape(xbar, (2, 1))
    theta = time1vars[2]
    dtheta = time2vars[2] - time1vars[2]
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    xbar = np.matmul(-R, xbar)
    T = np.identity(3)
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
    T[0:2, 0:2] = R
    T[0, 2] = xbar[0]
    T[1, 2] = xbar[1]

    return T

def extract_translation(T):
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    t = np.reshape(t, (2, 1))
    t = np.matmul(-R.transpose(), t)
    return t[0], t[1]

def remove_unlikely_data(counts, dt, dr, min_remove=10):
    remove = 0
    for i in range(counts.shape[0] - 1, 0, -1):
        if counts[i] < min_remove:
            remove += 1
        else:
            break
    if remove > 0:
        counts = counts[:-remove]
        dt = dt[:-remove]
        dr = dr[:-remove]
    return counts, dt, dr

if __name__ == '__main__':
    speed = []
    omega = []
    dt_rigid = []
    dr_rigid = []
    dt_md = []
    dr_md = []
    dt_dopp = []
    dr_dopp = []

    # gpsfile = '/home/keenan/radar_ws/data/2019-01-10-14-36-48-radar-oxford-10k-partial/gps/ins.csv'

    outlier_count = 0

    threshold = 2.0

    afile = 'accuracy.csv'
    if len(sys.argv) > 1:
        afile = sys.argv[1]

    print(afile)

    with open(afile, 'r') as f:
        f.readline()
        for line in f:
            row = line.split(',')
            dtr = np.sqrt((float(row[3]) - float(row[0]))**2 + (float(row[4]) - float(row[1]))**2)
            dtmd = np.sqrt((float(row[3]) - float(row[8]))**2 + (float(row[4]) - float(row[9]))**2)
            # dtd = np.sqrt((float(row[3]) - float(row[11]))**2 + (float(row[4]) - float(row[12]))**2)
            if dtr > threshold or dtmd > threshold: # or dtd > threshold:
                outlier_count += 1
                continue

            dt_rigid.append(dtr)
            dr_rigid.append(180 * abs(float(row[5]) - float(row[2])) / np.pi)
            dt_md.append(dtmd)
            dr_md.append(180 * abs(float(row[5]) - float(row[10])) / np.pi)
            # dt_dopp.append(dtd)
            # dr_dopp.append(180 * abs(float(row[5]) - float(row[13])) / np.pi)
            time1 = float(row[6])
            time2 = float(row[7])
            delta_t = (time2 - time1) / 1000000.0
            speed.append(np.sqrt(float(row[3])**2 + float(row[4])**2) / delta_t)
            omega.append(abs(float(row[5])) / delta_t)

    print('outliers: {} / {}'.format(outlier_count, len(speed)))

    dt_rigid = np.array(dt_rigid)
    dr_rigid = np.array(dr_rigid)
    dt_md = np.array(dt_md)
    dr_md = np.array(dr_md)
    # dt_dopp = np.array(dt_dopp)
    # dr_dopp = np.array(dr_dopp)
    speed = np.array(speed)
    omega = np.array(omega)

    print('RIGID: dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt_rigid), np.mean((dt_rigid - np.median(dt_rigid))**2), np.median(dr_rigid), np.mean((dr_rigid - np.median(dr_rigid))**2)))
    print('MDRANSAC: dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt_md), np.mean((dt_md - np.median(dt_md))**2), np.median(dr_md), np.mean((dt_md - np.median(dt_md))**2)))
    # print('MDRANSAC + DOPPLER: dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt_dopp), np.mean((dt_dopp - np.median(dt_dopp))**2), np.median(dr_dopp), np.mean((dt_dopp - np.median(dt_dopp))**2)))

    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
    axs[0].set_title('Translation Error (m) vs. Speed (m/s)')
    axs[1].set_title('Rotation Error (deg) vs. Speed (m/s)')

    max_speed = int(np.ceil(max(speed)))

    t_bins = np.zeros((max_speed))
    r_bins = np.zeros((max_speed))
    counts = np.zeros((max_speed))
    for j in range(0, speed.shape[0]):
        bin = int(np.floor(speed[j]))
        t_bins[bin] += dt_rigid[j]
        r_bins[bin] += dr_rigid[j]
        counts[bin] += 1
    counts, t_bins, r_bins = remove_unlikely_data(counts, t_bins, r_bins)
    t_bins /= counts
    r_bins /= counts

    axs[0].plot(t_bins, 'ob-', label='RIGID', linewidth=2)
    axs[1].plot(r_bins, 'ob-', label='RIGID', linewidth=2)

    t_bins = np.zeros((max_speed))
    r_bins = np.zeros((max_speed))
    counts = np.zeros((max_speed))
    for j in range(0, speed.shape[0]):
        bin = int(np.floor(speed[j]))
        t_bins[bin] += dt_md[j]
        r_bins[bin] += dr_md[j]
        counts[bin] += 1
    counts, t_bins, r_bins = remove_unlikely_data(counts, t_bins, r_bins)
    t_bins /= counts
    r_bins /= counts
    axs[0].plot(t_bins, 'or-', label='MDRANSAC', linewidth=2)
    axs[1].plot(r_bins, 'or-', label='MDRANSAC', linewidth=2)

    # t_bins = np.zeros((max_speed))
    # r_bins = np.zeros((max_speed))
    # counts = np.zeros((max_speed))
    # for j in range(0, speed.shape[0]):
    #     bin = int(np.floor(speed[j]))
    #     t_bins[bin] += dt_dopp[j]
    #     r_bins[bin] += dr_dopp[j]
    #     counts[bin] += 1
    #
    # t_bins /= counts
    # r_bins /= counts
    # counts, t_bins, r_bins = remove_unlikely_data(counts, t_bins, r_bins)
    # axs[0].plot(t_bins, 'og-', label='MDRANSAC+Doppler', linewidth=2)
    # axs[1].plot(r_bins, 'og-', label='MDRANSAC+Doppler', linewidth=2)

    for item in ([axs[0].title, axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
        item.set_fontsize(15)
    for item in ([axs[1].title, axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        item.set_fontsize(15)

    axLine, axLabel = axs[0].get_legend_handles_labels()
    fig.legend(axLine, axLabel, bbox_to_anchor=(0.20, 0.90), fontsize='x-small')
    # plt.show()

    plt.savefig('accuracy_vs_speed.png')

    # Error vs. Angular Velocity

    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
    axs[0].set_title('Translation Error (m) vs. Ang Vel (rad/s)')
    axs[1].set_title('Rotation Error (deg) vs. Ang Vel (rad/s)')

    max_omega = max(omega)
    num_bins = 10
    omegabins = np.arange(0, max_omega, step = max_omega / num_bins)


    def get_bin(x, highest, num_bins):
        bin_width = highest / num_bins
        bin = int(np.floor(abs(x) / bin_width))
        if bin >= num_bins:
            bin = num_bins - 1
        return bin

    t_bins = np.zeros((num_bins))
    r_bins = np.zeros((num_bins))
    counts = np.zeros((num_bins))
    for j in range(0, omega.shape[0]):
        bin = get_bin(omega[j], max_omega, num_bins)
        t_bins[bin] += dt_rigid[j]
        r_bins[bin] += dr_rigid[j]
        counts[bin] += 1
    print(counts)
    # counts, t_bins, r_bins = remove_unlikely_data(counts, t_bins, r_bins)
    t_bins /= counts
    r_bins /= counts

    print(t_bins.shape)
    print(omegabins.shape)

    axs[0].plot(omegabins, t_bins, 'ob-', label='RIGID', linewidth=2)
    axs[1].plot(omegabins, r_bins, 'ob-', label='RIGID', linewidth=2)

    t_bins = np.zeros((num_bins))
    r_bins = np.zeros((num_bins))
    counts = np.zeros((num_bins))
    for j in range(0, omega.shape[0]):
        bin = get_bin(omega[j], max_omega, num_bins)
        t_bins[bin] += dt_md[j]
        r_bins[bin] += dr_md[j]
        counts[bin] += 1
    # counts, t_bins, r_bins = remove_unlikely_data(counts, t_bins, r_bins)
    t_bins /= counts
    r_bins /= counts

    axs[0].plot(omegabins, t_bins, 'or-', label='MDRANSAC', linewidth=2)
    axs[1].plot(omegabins, r_bins, 'or-', label='MDRANSAC', linewidth=2)

    # Create plot of the trajectories
    T_gt = np.identity(3)
    T_rigid = np.identity(3)
    T_md = np.identity(3)
    # T_gps = np.identity(3)
    T_dopp = np.identity(3)

    xgt = []
    ygt = []
    xrigid = []
    yrigid = []
    xmd = []
    ymd = []
    xgps = []
    ygps = []
    xdopp = []
    ydopp = []

    t_error_rigid = []
    t_error_md = []

    with open(afile) as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            # Create transformation matrices
            T_gt_ = get_transform(float(row[3]), float(row[4]), float(row[5]))
            T_rigid_ = get_transform(float(row[0]), float(row[1]), float(row[2]))
            T_md_ = get_transform(float(row[8]), float(row[9]), float(row[10]))
            T_dopp_ = get_transform(float(row[11]), float(row[12]), float(row[13]))
            T_gt = np.matmul(T_gt, T_gt_)
            T_rigid = np.matmul(T_rigid, T_rigid_)
            T_md = np.matmul(T_md, T_md_)
            T_dopp = np.matmul(T_dopp, T_dopp_)

            R_gt = T_gt[0:2,0:2]
            R_rigid = T_rigid[0:2,0:2]
            R_md = T_md[0:2,0:2]
            R_dopp = T_dopp[0:2,0:2]
            if np.linalg.det(R_gt) != 1.0:
                enforce_orthogonality(R_gt)
                T_gt[0:2,0:2] = R_gt
            if np.linalg.det(R_rigid) != 1.0:
                enforce_orthogonality(R_rigid)
                T_rigid[0:2,0:2] = R_rigid
            if np.linalg.det(R_md) != 1.0:
                enforce_orthogonality(R_md)
                T_md[0:2,0:2] = R_md
            if np.linalg.det(R_dopp) != 1.0:
                enforce_orthogonality(R_dopp)
                T_dopp[0:2,0:2] = R_dopp

            pose_error = np.matmul(get_inverse_tf(T_rigid), T_gt)
            t_error_rigid.append(translationError(pose_error))
            pose_error2 = np.matmul(get_inverse_tf(T_md), T_gt)
            t_error_md.append(translationError(pose_error2))

            # Get GPS ground truth between the frames
            # time1 = int(row[6])
            # time2 = int(row[7])
            # T_gps_ = get_ins_transformation(time1, time2, gpsfile)
            # T_gps = np.matmul(T_gps, T_gps_)
            # R_gps = T_gps[0:2,0:2]
            # if np.linalg.det(R_gps) != 1.0:
            #     enforce_orthogonality(R_gps)
            #     T_gps[0:2,0:2] = R_gps

            xgt.append(T_gt[0, 2])
            ygt.append(T_gt[1, 2])
            xrigid.append(T_rigid[0, 2])
            yrigid.append(T_rigid[1, 2])
            xmd.append(T_md[0, 2])
            ymd.append(T_md[1, 2])
            # xgps.append(T_gps[0, 2])
            # ygps.append(T_gps[1, 2])
            xdopp.append(T_dopp[0, 2])
            ydopp.append(T_dopp[1, 2])

    xgt = np.array(xgt)
    ygt = np.array(ygt)
    rgt = [0]
    for i in range(1, xgt.shape[0]):
        delta_x = xgt[i] - xgt[i - 1]
        delta_y = ygt[i] - ygt[i - 1]
        rgt.append(np.sqrt(delta_x **2 + delta_y **2) + rgt[i - 1])

    rgt = np.array(rgt)
    xrigid = np.array(xrigid)
    yrigid = np.array(yrigid)
    xmd = np.array(xmd)
    ymd = np.array(ymd)
    # xgps = np.array(xgps)
    # ygps = np.array(ygps)
    xdopp = np.array(xdopp)
    ydopp = np.array(ydopp)

    matplotlib.rcParams.update({'font.size': 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16,
                                'axes.linewidth' : 1.5, 'font.family' : 'serif', 'pdf.fonttype' : 42})

    plt.figure(figsize=(10, 5))
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.axes().set_aspect('equal')
    plt.plot(xgt, ygt, 'k', linewidth=2.5, label='Ground Truth')
    plt.plot(xrigid, yrigid, 'r', linewidth=2.5, label='RIGID')
    plt.plot(xmd, ymd, 'b', linewidth=2.5, label='MC-RANSAC')
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('y (m)', fontsize=16)
    plt.legend(loc="upper left")
    plt.savefig('trajectory.pdf', bbox_inches='tight', pad_inches=0.0)

    # plt.figure(figsize=(10, 5))
    # plt.grid(which='both', linestyle='--', alpha=0.5)
    # t_error_rigid = np.array(t_error_rigid)
    # t_error_md = np.array(t_error_md)
    # plt.plot(rgt, t_error_rigid, 'r', linewidth=2.5, label='BASELINE')
    # plt.plot(rgt, t_error_md, 'b', linewidth=2.5, label='MOTION\nCOMPENSATED')
    # plt.xlabel('Distance Traveled (m)', fontsize=16)
    # plt.ylabel('Error (m)', fontsize=16)
    # plt.legend(loc="upper left")
    # plt.savefig('traj_error.pdf', bbox_inches='tight', pad_inches=0.0)
