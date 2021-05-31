import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from get_radar_loc_gt import yaw, rotToYawPitchRoll

def getRotDiff(r1, r2):
    C1 = yaw(r1)
    C2 = yaw(r2)
    C_err = np.matmul(C2.transpose(), C1)
    yaw_err, _, _ = rotToYawPitchRoll(C_err)
    return abs(yaw_err)

if __name__ == "__main__":
    file = "localization_accuracy_icra4.csv"

    dt1 = []
    dt2 = []
    dt3 = []
    dt4 = []
    dt5 = []
    dr1 = []
    dr2 = []
    dr3 = []
    dr4 = []
    dr5 = []

    with open(file, 'r') as f:
        f.readline()
        for line in f:
            row = line.split(',')
            gtx = float(row[15])
            gty = float(row[16])
            gtyaw = float(row[17])

            dt1.append(np.sqrt((gtx - float(row[0]))**2 + (gty - float(row[1]))**2))
            dr1.append(180 * getRotDiff(gtyaw, float(row[2])) / np.pi)

            dt2.append(np.sqrt((gtx - float(row[3]))**2 + (gty - float(row[4]))**2))
            dr2.append(180 * getRotDiff(gtyaw, float(row[5])) / np.pi)

            dt3.append(np.sqrt((gtx - float(row[6]))**2 + (gty - float(row[7]))**2))
            dr3.append(180 * getRotDiff(gtyaw, float(row[8])) / np.pi)

            dt4.append(np.sqrt((gtx - float(row[9]))**2 + (gty - float(row[10]))**2))
            dr4.append(180 * getRotDiff(gtyaw, float(row[11])) / np.pi)

            dt5.append(np.sqrt((gtx - float(row[12]))**2 + (gty - float(row[13]))**2))
            dr5.append(180 * getRotDiff(gtyaw, float(row[14])) / np.pi)

    dt1 = np.array(dt1)
    dt2 = np.array(dt2)
    dt3 = np.array(dt3)
    dt4 = np.array(dt4)
    dt5 = np.array(dt5)
    dr1 = np.array(dr1)
    dr2 = np.array(dr2)
    dr3 = np.array(dr3)
    dr4 = np.array(dr4)
    dr5 = np.array(dr5)

    np.savetxt('dr3', dr3)

    print('RIGID: dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt1), np.mean((dt1 - np.median(dt1))**2), np.median(dr1), np.mean((dr1 - np.median(dr1))**2)))
    print('DOPP ONLY: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt2), np.mean((dt2 - np.median(dt2))**2), np.median(dr2), np.mean((dr2 - np.median(dr2))**2)))
    print('DOPP + MD: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt3), np.mean((dt3 - np.median(dt3))**2), np.median(dr3), np.mean((dr3 - np.median(dr3))**2)))
    print('MD ONLY: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt4), np.mean((dt4 - np.median(dt4))**2), np.median(dr4), np.mean((dr4 - np.median(dr4))**2)))
    print('MD + DOPP: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt5), np.mean((dt5 - np.median(dt5))**2), np.median(dr5), np.mean((dr5 - np.median(dr5))**2)))

    matplotlib.rcParams.update({"font.size" : 16, 'xtick.labelsize' : 16, 'ytick.labelsize' : 16,
                                'axes.linewidth' : 1.5, 'font.family' : 'serif', 'pdf.fonttype' : 42})
    plt.figure(figsize=(10, 5.5))
    bins = np.arange(0, 4.0, 0.25)
    plt.grid(which='both', linestyle='--', alpha=0.5, axis='y')
    plt.hist([dt1, dt4, dt3], bins=bins, label=['RIGID', 'MC', 'MC+Dopp'], color=['r', 'b', 'limegreen'], rwidth=0.6)
    plt.xlabel('Translation Error (m)', fontsize=18)
    plt.ylabel('Number of Radar Pairs', fontsize=18)
    plt.legend(loc='best')
    plt.savefig('localization_accuracy.pdf', bbox_inches='tight', pad_inches=0.0)
    # plt.show()
