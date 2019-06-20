import psycopg2
import time
import math
import numpy as np
import sys
import os
import random
import datetime
import random
import pycurl
import StringIO
import npVas as vas
import json

res_x = 1920 / 4
res_y = 1080 / 4

y0 = 21.146359
y1 = 49.012657

x1 = -74.349531
x0 = -126.391414

us_map = ((-170, -60), (15, 70))
ny_map = ((-74.129, -71.8298), (40.3153, 41.1069))
map = us_map

rv = (32, 18)
ev = (480, 270)
hv = (100, 100)

yStep = (y1 - y0) / res_y
xStep = (x1 - x0) / res_x

##############Tweets ID########################################################
maxID = 1009129304360288256
minID = 862500001267011588
interval_size = 293258606186553  # 500
startID = 862500001267011588
#############################################################################
##############Uber ID########################################################
# maxID=1412117940
# minID=1396310400
# startID=1396310400
#############################################################################

#############################################################################
##############Texi ID########################################################
# maxID=999999996
# minID=16
# startID=16
#############################################################################

# postgresql connection
conStr = "dbname='postgres' user='postgres' host='166.111.71.132' port='5432' password='postgres' "
conn = psycopg2.connect(conStr)
cur = conn.cursor()


# oracle connection
# ora_conn=cx_Oracle.connect("system","Oracle123","curium.ics.uci.edu:1521/orcl")
# ora_cur=ora_conn.cursor()

# Map the coodrinates into cells, the type of 'ar' is the numpy array, r is the coordinate range of the map. the returned value H is the matrix of cells,
# each value is the number of records in the cell.
def hashByNumpy(ar, r=map, b=(res_x, res_y)):
    H, x, y = np.histogram2d(ar[:, 0], ar[:, 1], bins=b, range=r)
    return H


def imageLen(array):
    return np.count_nonzero(hashByNumpy(array))


# return the mse of two matrix
def myMSE(m1, m2, binary=True):  # m1, m2 are the matrixs of the ground-truth map and approximate map
    if binary:
        m1 = np.where(m1 > 0, 1, 0)  # convert each element to 0 or 1
        m2 = np.where(m2 > 0, 1, 0)  # convert each element to 0 or 1
    err = 0
    for i in range(0, len(m1)):
        for j in range(0, len(m1[0])):
            err += (m1[i][j] - m2[i][j]) ** 2
    return math.sqrt(err) / (len(m1) * len(m1[0]))


# Find #records and ratio in random ranges.
def RangeMerge(tab='biasedtweets'):
    maxID = 1009129304360288256
    minID = 862500001267011588
    step = (maxID - minID) / 1000
    for bin in range(0, 1):
        startID = minID + step * 10 * 12 * bin
        endID = startID + step * 10 * 12
        # sql="select coordinate[0],coordinate[1] from "+tab+" where id>="+str(startID)+" and id<"+str(endID)
        sql = "select coordinate[0],coordinate[1] from " + tab + " limit 5000000 offset " + str(
            bin * 10000000 + 20000000)
        cur.execute(sql)
        coord = np.array(cur.fetchall())
        if len(coord > 0):
            pLen = imageLen(coord)
            print 'Group', 'Ratio', '#Record', '#Point', '#SubsetRecord', '#SubsetPoint', 'Quality' + str(bin)
            for r in range(1, 11):
                k = int(float(r) / 10.0 * len(coord))
                sLen = imageLen(coord[:k])
                print bin, r * 10, len(coord), pLen, k, sLen, float(sLen) / pLen


def clear_histogram(tab=""):
    sql = "delete from " + tab
    cur.execute(sql)
    cur.execute("commit")


def dividing_points(nEV, point_num):
    id = minID
    intervalSize = (maxID - minID) / nEV
    for i in range(0, point_num):
        sql = "insert into splitting_point values(" + str(id) + ",0,0,0)"
        cur.execute(sql)
        cur.execute("commit")
        id = id + intervalSize


def create_histogram(nEV, tab, nBucket=10, nInterval=10):
    sql = "select point from splitting_point order by point asc"
    cur.execute(sql)
    points = cur.fetchall()
    startPoint = points[0]
    parent_id = 0.0
    for endPoint in points:
        # retrieve data
        sql = "select coordinate[0],coordinate[1] from " + tab + " where id between " + str(
            startPoint[0]) + " and " + str(endPoint[0])
        cur.execute(sql)
        coord = np.array(cur.fetchall())
        if len(coord) < 1:
            continue
        # a new parent intval
        sql = "insert into parent_interval(parent_id,startval,endval) values(" + str(parent_id) + "," + str(
            startPoint[0]) + "," + str(endPoint[0]) + ")"
        cur.execute(sql)
        cur.execute("commit")
        # records for the new parent_interval
        OriginalViz = hashByNumpy(coord, r=map, b=ev)
        for x in range(0, ev[0]):
            for y in range(0, ev[1]):
                if OriginalViz[x][y] != 0:
                    sql = "insert into parent_pixels(parent_id,x,y) values(" + str(parent_id) + "," + str(
                        x) + "," + str(y) + ")"
                    cur.execute(sql)
        cur.execute("commit")
        # child intervals
        for r in range(0, nBucket):
            ks = int(r * 10 / 100.0 * len(coord))
            ke = int((r + 1) * 10 / 100.0 * len(coord))
            tmpVizA = hashByNumpy(coord[0:ke], r=map, b=ev)
            LowVizA = hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizA))), r=((0, ev[0]), (0, ev[1])), b=rv)
            ###reversed value
            tmpVizB = hashByNumpy(coord[ks - len(coord):], r=map, b=ev)
            LowVizB = hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizB))), r=((0, ev[0]), (0, ev[1])), b=rv)

            for x in range(0, rv[0]):
                for y in range(0, rv[1]):
                    if LowVizA[x][y] != 0 or LowVizB[x][y] != 0:
                        sql = "insert into child_interval(parent_id,child_id,x,y,a,b) values(" + str(
                            parent_id) + "," + str(r) + "," + str(x) + "," + str(y) + "," + str(
                            LowVizA[x][y]) + "," + str(LowVizB[x][y]) + ")"
                        cur.execute(sql)
            cur.execute("commit")
        startPoint = endPoint
        parent_id += 2048.0
    print "DONE."


def SnapShot(dt, rdt, nEV, nInterval, nBucket, tab):
    step = (maxID - minID) / nInterval
    for i in range(0, nEV):
        sql = "select coordinate[0],coordinate[1] from " + tab + " where id>=" + str(
            startID + i * step) + " and id<" + str(startID + (i + 1) * step)

        qs = time.time()
        cur.execute(sql)
        coord = np.array(cur.fetchall())
        qe = time.time()

        if len(coord) < 1:
            continue

        ps = time.time()
        OriginalViz = hashByNumpy(coord, r=map, b=ev)
        pe = time.time()

        ss = time.time()
        for x in range(0, ev[0]):
            for y in range(0, ev[1]):
                if OriginalViz[x][y] != 0:
                    sql = "insert into " + dt + " values(" + str(i) + "," + str(x) + "," + str(y) + ")"
                    cur.execute(sql)
        cur.execute("commit")
        se = time.time()
        # print "Original viz for partition",i,"Done."

        ############################################
        rt = 0
        for r in range(0, nBucket):
            ks = int(r * 10 / 100.0 * len(coord))
            ke = int((r + 1) * 10 / 100.0 * len(coord))
            rs = time.time()
            tmpVizA = hashByNumpy(coord[0:ke], r=map, b=ev)
            LowVizA = hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizA))), r=((0, ev[0]), (0, ev[1])), b=rv)
            ###reversed value
            tmpVizB = hashByNumpy(coord[ks - len(coord):], r=map, b=ev)
            LowVizB = hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizB))), r=((0, ev[0]), (0, ev[1])), b=rv)
            re = time.time()
            rt += re - rs

            for x in range(0, rv[0]):
                for y in range(0, rv[1]):
                    if LowVizA[x][y] != 0 or LowVizB[x][y] != 0:
                        sql = "insert into " + rdt + " values(" + str(i) + "," + str(r) + "," + str(x) + "," + str(
                            y) + "," + str(LowVizA[x][y]) + "," + str(LowVizB[x][y]) + ")"
                        cur.execute(sql)
            cur.execute("commit")
            # print "RV for partition",i,"bucket",r, ", Done."
        print qe - qs, pe - ps + rt, se - ss
    print "DONE."


def RewrittenQuery(lev, rev, yl, yr, xl, xr, etab, rtab):  # 14963
    sql = "select distinct x,y from " + etab + " where viz>=" + str(lev) + " and viz<=" + str(rev)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_rl = hashByNumpy(np.array(hist), r=((0, ev[0]), (0, ev[1])), b=rv)

    sql = "select distinct x,y from " + etab + " where viz>" + str(lev) + " and viz<" + str(rev)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_m = np.zeros(shape=rv)
    if len(hist) > 0:
        RV_m = hashByNumpy(np.array(hist), r=((0, ev[0]), (0, ev[1])), b=rv)

    sql = "select x,y,b from " + rtab + " where part=" + str(lev) + " and buck=" + str(yl)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_yl = np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist, RV_yl)

    sql = "select x,y,a from " + rtab + " where part=" + str(rev) + " and buck=" + str(yr)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_yr = np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist, RV_yr)

    sql = "select x,y,b from " + rtab + " where part=" + str(lev) + " and buck=" + str(xl)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_xl = np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist, RV_xl)

    sql = "select x,y,a from " + rtab + " where part=" + str(rev) + " and buck=" + str(xr)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_xr = np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist, RV_xr)

    numerator = 0.0
    denominator = 0.0

    for i in range(0, rv[0]):
        for j in range(0, rv[1]):
            tmp = max(RV_xl[i][j], RV_xr[i][j], RV_m[i][j])
            numerator += tmp
            denominator += min(tmp + RV_yl[i][j] - RV_xl[i][j] + RV_yr[i][j] - RV_xr[i][j], RV_rl[i][j])

    return numerator / denominator


def SSIM(X, Y):
    x = X.flatten()
    y = Y.flatten()
    ux = np.mean(x)
    uy = np.mean(y)
    sigmax = np.cov(x)
    sigmay = np.cov(y)
    sigmaxy = np.cov(x, y)[0, 1]
    ##
    k1 = 0.01
    k2 = 0.03
    L = 1
    ##
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    ##
    l = (2 * ux * uy + C1) / (ux ** 2 + uy ** 2 + C1)
    c = (2 * math.sqrt(sigmax) * math.sqrt(sigmay) + C2) / (sigmax + sigmay + C2)
    s = (sigmaxy + C3) / (math.sqrt(sigmax) * math.sqrt(sigmay) + C3)
    ##
    return l * c * s


def BiasedQueries(minID=1396310400, maxID=1412117940, nInterval=20):
    size = (maxID - minID) / nInterval
    for s in range(1, nInterval - 10):
        for i in range(0, 100):
            ID1 = random.randint(minID + size * 10, minID + size * 11)
            hisA = IntervalChk(ID=ID1)
            hisB = IntervalChk(ID=ID1 + s * size)
            print ID1, ID1 + s * size, hisA[0], hisA[1], hisB[0], hisB[1]


# generate queries of length, 1st group: start: 1-3,end:3-5 , 2nd group: start:5-7, end: 7-9
def random_ranges(nInterval):
    f = open("skewed_queries.txt", "ab+")
    intervalSize = (maxID - minID) / nInterval
    for i in range(0, 1000):
        start = random.randint(minID + intervalSize, minID + 3 * intervalSize)
        end = start + int(intervalSize * random.uniform(0.5, 2.0))
        f.write(str(start) + " " + str(end) + "\n")

    for i in range(0, 1000):
        start = random.randint(minID + 5 * intervalSize, minID + 7 * intervalSize)
        end = start + int(intervalSize * random.uniform(1.0, 2.0))
        f.write(str(start) + " " + str(end) + "\n")


def RandomQueries(nInterval, fn="random_ranges.txt"):
    intervalSize = (maxID - minID) / nInterval
    f = open(fn, 'ab+')
    for length in range(1, 11):
        for i in range(0, 10):
            start = random.randint(minID, minID + intervalSize)
            end = int(start + length * intervalSize)
            len = math.floor((end - start) / intervalSize)
            f.write(str(start) + " " + str(end) + " " + str(len) + '\n')


def RandomQuereisForVAS(nInterval, fn):
    f = open(fn, "ab+")
    intervalSize = (maxID - minID) / nInterval
    for i in range(1, 7):
        for j in range(0, 2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end = start + 3 * intervalSize
            f.write(str(start) + " " + str(end) + '\n')
    for i in range(1, 5):
        for j in range(0, 2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end = start + 5 * intervalSize
            f.write(str(start) + " " + str(end) + '\n')
    for i in range(1, 3):
        for j in range(0, 2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end = start + 7 * intervalSize
            f.write(str(start) + " " + str(end) + '\n')
    for i in range(1, 2):
        for j in range(0, 2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end = start + 9 * intervalSize
            f.write(str(start) + " " + str(end) + '\n')


def GetIntervals(nInterval, fn="random_ranges.txt", dstFn='random_workload.txt'):
    fo = open(fn)
    f = open(dstFn, "ab+")
    for line in fo.readlines():
        ids = line.split()
        hisA = IntervalChk(nInterval, 10, int(ids[0]))
        hisB = IntervalChk(nInterval, 10, int(ids[1]))
        f.write(ids[0] + " " + ids[1] + " " + str(hisA[0]) + " " + str(hisA[1]) + " " + str(hisB[0]) + " " + str(
            hisB[1]) + "\n")


def IntervalChk(nInterval, sInterval=10, ID=0):
    IntervalLen = (maxID - minID) / nInterval
    sIntervalLen = IntervalLen / sInterval

    IntervalID = math.floor((ID - minID) / IntervalLen)
    sIntervalID = math.floor((ID - minID - IntervalID * IntervalLen) / sIntervalLen)

    return int(IntervalID), int(sIntervalID)


# comparison of quality functions: jaccard, ssim, mse


def QualityFunctions():
    coord = np.array(GetCoordinateUber('bigtweets', '*', limit=5000000))
    gt = hashByNumpy(coord, r=map)
    gt = np.where(gt > 0, 1, 0)  # convert each element to 0 or 1
    for r in range(0, 101, 10):
        aprmt = hashByNumpy(coord[:int(len(coord) * r / 100.0)], r=map)
        aprmt = np.where(aprmt > 0, 1, 0)
        print np.count_nonzero(aprmt) / float(np.count_nonzero(gt)), lmt.SSIM(gt, aprmt), lmt.myMSE(gt, aprmt)


def quality_alpha_theta(startEV, endEV, tab="EV"):
    sql = "select distinct x,y from " + tab + " where viz>=" + str(startEV) + " and viz<=" + str(endEV)
    cur.execute(sql)
    hist1 = cur.fetchall()
    sql = "select distinct x,y from " + tab + " where viz>" + str(startEV) + " and viz<" + str(endEV)
    cur.execute(sql)
    hist2 = cur.fetchall()
    if len(hist1) > 0 and len(hist2) > 0:
        quality = len(hist2) / float(len(hist1))
        return quality
    else:
        return 0


def CheckIntervalHead(RV, EV, IntervalFrac=70):
    parentIntervalSize = (maxID - minID) / IntervalFrac
    childIntervalSize = parentIntervalSize / 10
    endID = minID + (EV + 1) * parentIntervalSize
    startID = endID - RV * childIntervalSize
    return startID, endID


def CheckIntervalTail(RV, EV, IntervalFrac=70):
    parentIntervalSize = (maxID - minID) / IntervalFrac
    childIntervalSize = parentIntervalSize / 10
    startID = minID + EV * parentIntervalSize
    endID = startID + RV * childIntervalSize
    return startID, endID


def quality_alpha_gamma(startEV, endEV, startRV, endRV, etab, rtab):
    quality = RewrittenQuery(startEV, endEV, startRV, endRV, -1, -1, etab, rtab)
    return quality


def quality_alpha_beta_gamma(startEV, endEV, startRV, endRV, tau, etab, rtab):
    alpha_theta = quality_alpha_theta(startEV, endEV, etab)
    alpha_gamma = quality_alpha_gamma(startEV, endEV, startRV, endRV, etab, rtab)
    ##
    quality = 0.0
    ##if starts and ends at the same parent interval, then we only need to search from the startRV+1 and ends at the endRV
    if startEV == endEV:
        for xr in range(startRV + 1, endRV):
            quality = RewrittenQuery(startEV, endEV, startRV, endRV, -1, xr, etab, rtab)
            if quality >= tau:
                return alpha_theta, alpha_gamma, quality, -1, xr
    ##else we use different search range.
    else:
        for xr in range(0, endRV):
            quality = RewrittenQuery(startEV, endEV, startRV, endRV, -1, xr, etab, rtab)
            if quality >= tau:
                return alpha_theta, alpha_gamma, quality, -1, xr
        for xl in range(9, startRV, -1):
            quality = RewrittenQuery(startEV, endEV, startRV, endRV, xl, endRV - 1, etab, rtab)
            if quality >= tau:
                return alpha_theta, alpha_gamma, quality, xl, endRV - 1
    return alpha_theta, alpha_gamma, quality, startRV + 1, endRV - 1


def three_qualities(tau, fn, resFile, etab, rtab):
    file = open(fn)
    resFn = open(resFile, "ab+")
    for line in file.readlines():
        strs = line.split()
        startEV = int(strs[2])
        startRV = int(strs[3])
        endEV = int(strs[4])
        endRV = int(strs[5])
        result = quality_alpha_beta_gamma(startEV, endEV, startRV, endRV, tau, etab, rtab)
        resFn.write(
            strs[0] + " " + strs[1] + " " + str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + strs[
                2] + " " + str(result[3]) + " " + strs[4] + " " + str(result[4]) + "\n")


def alpha2theta(fn, tab, resFile):
    file = open(fn)
    resFn = open(resFile, "ab+")
    for line in file.readlines():
        strs = line.split()
        startEV = int(strs[2])
        endEV = int(strs[4])
        resFn.write(strs[0] + " " + strs[1] + " " + strs[2] + " " + strs[4] + " " + str(
            quality_alpha_theta(startEV, endEV, tab)) + "\n")


def record_count(fn, resFile):
    f = open(fn)
    resFn = open(resFile, "ab+")
    i = 0
    for line in f.readlines():
        ids = line.split()
        sql = "select count(*) from bigtweets where id between " + ids[0] + " and " + ids[1]
        cur.execute(sql)
        left = cur.fetchall()[0][0]

        sql = "select count(*) from bigtweets where id between " + ids[2] + " and " + ids[3]
        cur.execute(sql)
        right = cur.fetchall()[0][0]
        resFn.write(str(int(left) + int(right)) + "\n")
        print int(left) + int(right)
        i += 1
        if i == 999:
            return


def QualityDiff():
    IntervalSize = (maxID - minID) / 70
    for length in range(1, 6):
        start = int(1.1 * IntervalSize + startID)
        end = start + length * IntervalSize
        coord = GetCoordinateRange('taxi', start, end)
        OriginalLen = imageLen(np.array(coord))
        for r in range(0, 10):
            subLen = imageLen(np.array(coord[:int(len(coord) / (length * 10.0) * ((length - 1) * 10.0 + r + 1))]))
            print float(subLen) / OriginalLen
        for r in range(0, 1):
            print RewrittenQuery(1, 2 + length - 1, 1, 0, -1, r)
        for r in range(9, 0, -1):
            print RewrittenQuery(1, 2 + length - 1, 1, 0, r, 0)


def getIntervalTail(id, nInterval):
    intervalSize = (maxID - minID) / nInterval
    return minID + (id + 1) * intervalSize


def getIntervalHead(id, nInterval):
    intervalSize = (maxID - minID) / nInterval
    return minID + id * intervalSize


def getChildIntervalHead(parentID, childID, nInterval):
    intervalSize = (maxID - minID) / nInterval
    childIntervalSize = intervalSize / 10
    return minID + parentID * intervalSize + childID * childIntervalSize


def getChildIntervalTail(parentID, childID, nInterval):
    intervalSize = (maxID - minID) / nInterval
    childIntervalSize = intervalSize / 10
    return minID + parentID * intervalSize + (childID + 1) * childIntervalSize


def result_size_ev(nInterval, fn, resFile):
    f = open(fn)
    resFn = open(resFile, "ab+")
    for line in f.readlines():
        evLL = -1
        evLR = -1
        evRL = -1
        evRR = -1
        strs = line.split()
        if float(strs[4]) < 0.85:
            evLL = strs[0]
            evLR = getIntervalTail(int(strs[2]), nInterval)
            evRL = getIntervalHead(int(strs[3]), nInterval)
            evRR = strs[1]
            # check if interval overlaps or exceeds
        if evRL <= evLR:
            evLR = evRR
            evRL = -1
            evRR = -1
        resFn.write(str(evLL) + " " + str(evLR) + " " + str(evRL) + " " + str(evRR) + "\n")


def result_size_evrv(nInterval, fn="", resFile="", tau=0.85):
    f = open(fn)
    resFn = open(resFile, "ab+")
    for line in f.readlines():
        rvLL = -1
        rvLR = -1
        rvRL = -1
        rvRR = -1
        strs = line.split()
        if float(strs[4]) < tau:
            rvLL = strs[0]
            rvLR = getIntervalTail(int(strs[5]), nInterval)
            rvRL = getIntervalHead(int(strs[7]), nInterval)
            rvRR = strs[1]
        elif float(strs[2]) < tau and float(strs[4]) >= tau:
            if int(strs[6]) != -1:
                rvLL = getChildIntervalHead(int(strs[5]), int(strs[6]), nInterval)
                rvLR = getIntervalTail(int(strs[5]), nInterval)
            if int(strs[8]) != -1:
                rvRL = getIntervalHead(int(strs[7]), nInterval)
                rvRR = getChildIntervalTail(int(strs[7]), int(strs[8]), nInterval)
            # check if interval overlaps or exceeds
            rvLR = min(rvLR, int(strs[1]))
            rvRL = max(rvRL, int(strs[0]))
        if rvRL <= rvLR:
            rvLR = rvRR
            rvRL = -1
            rvRR = -1

        resFn.write(str(rvLL) + " " + str(rvLR) + " " + str(rvRL) + " " + str(rvRR) + "\n")


def heatmap_histogram(dt, nEV, nInterval, tab):
    step = (maxID - minID) / nInterval
    for i in range(0, nEV):
        sql = "select coordinate[0],coordinate[1] from " + tab + " where id>=" + str(
            startID + i * step) + " and id<" + str(startID + (i + 1) * step)
        cur.execute(sql)
        coord = np.array(cur.fetchall())
        if len(coord) < 1:
            continue
        OriginalViz = hashByNumpy(coord, r=map, b=hv)
        for x in range(0, hv[0]):
            for y in range(0, hv[1]):
                # if OriginalViz[x][y]!=0:
                sql = "insert into " + dt + " values(" + str(i) + "," + str(x) + "," + str(y) + "," + str(
                    OriginalViz[x][y]) + ")"
                cur.execute(sql)
        cur.execute("commit")
    print "DONE."


def sample_seek(startID, endID):
    b = StringIO.StringIO()
    c = pycurl.Curl()
    sql = '{"select": ["x", "y"],"filters":[{"attribute": "id", "operator": "IN", "operands": [' + str(
        startID) + ', ' + str(endID) + ']}]}'
    c.setopt(pycurl.URL, 'http://localhost:8080/query')
    c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json', 'Content-Length:' + str(len(sql))])
    c.setopt(pycurl.CUSTOMREQUEST, "POST")
    c.setopt(pycurl.POSTFIELDS, sql)
    c.setopt(pycurl.WRITEFUNCTION, b.write)
    c.perform()
    c.close()
    val = json.loads(b.getvalue())
    b.close()
    return val["result"]


def sample_vas(nEV, nInterval):
    interval_size = (maxID - minID) / nInterval
    startID = minID
    endID = minID + interval_size * nEV

    sql = "select id,coordinate[0],coordinate[1],0,0 from bigtweets where id between " + str(startID) + " and " + str(
        endID)
    # sql="select id,x,y,0,0 from testVas"
    cur.execute(sql)
    sample = vas.VAS(cur.fetchall())
    for i in range(0, len(sample)):
        sql = "insert into vas_sample values(" + str(sample[i][0]) + ",point(" + str(sample[i][1]) + "," + str(
            sample[i][2]) + "))"
        cur.execute(sql)
    cur.execute("commit")

    print "VAS sample created."


def distribution_precision(startID, endID):
    coord1 = np.array(sample_seek(startID, endID))
    ss_matrix = hashByNumpy(coord1, map, hv)

    sql = "select coordinate[0],coordinate[1] from Bigtweets where ID between " + str(startID) + " and " + str(endID)
    cur.execute(sql)
    original_matrix = hashByNumpy(np.array(cur.fetchall()), map, hv)

    startHV = IntervalChk(500, 10, startID)[0]
    endHV = IntervalChk(500, 10, endID)[0]

    sql = "select x,y,sum(density) from HV where viz >" + str(startHV) + " and viz<" + str(endHV) + " group by x,y"
    cur.execute(sql)
    mvs_inner_list = np.array(cur.fetchall())

    sql = "select x,y,sum(density) from HV where viz >=" + str(startHV) + " and viz<=" + str(endHV) + " group by x,y"
    cur.execute(sql)
    mvs_outer_list = np.array(cur.fetchall())

    ss_sum = float(np.sum(ss_matrix))
    original_sum = float(np.sum(original_matrix))
    mvs_inner_sum = float(np.sum(mvs_inner_list[:, 2]))
    mvs_outer_sum = float(np.sum(mvs_outer_list[:, 2]))

    max_diff = 0.0
    max_index_x = 0
    max_index_y = 0
    max_index_i = 0
    for i in range(0, len(mvs_inner_list)):
        if mvs_outer_list[i][2] - mvs_inner_list[i][2] > max_diff:
            max_diff = mvs_outer_list[i][2] - mvs_inner_list[i][2]
            max_index_x = mvs_inner_list[i][0]
            max_index_y = mvs_inner_list[i][1]
            max_index_i = i

    ss_eps = 0.0
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ss_eps += (ss_matrix[i][j] / ss_sum - original_matrix[i][j] / original_sum) ** 2
    ss_eps = math.sqrt(ss_eps)

    mvs_real_eps = 0.0
    for row in mvs_inner_list:
        mvs_real_eps += (row[2] / mvs_inner_sum - original_matrix[row[0]][row[1]] / mvs_outer_sum) ** 2
    mvs_real_eps = math.sqrt(mvs_real_eps)

    mvs_est_eps = 0.0
    mvs_inner_sum += max_diff
    mvs_inner_sum -= mvs_inner_list[max_index_i][2]
    mvs_inner_list[max_index_i][2] = max_diff

    for i in range(0, len(mvs_inner_list)):
        mvs_est_eps += (mvs_inner_list[i][2] / mvs_inner_sum - mvs_outer_list[i][2] / original_sum) ** 2
    mvs_est_eps = math.sqrt(mvs_est_eps)
    print "sample seek precision:", ss_eps, "mvs real precision:", mvs_real_eps, "mvs est precision:", mvs_est_eps


def vas_quality(fn):
    f = open(fn)
    for line in f.readlines():
        id = line.split()
        sql = "select coordinate[0],coordinate[1] from bigtweets where id between " + id[0] + " and " + id[1]
        cur.execute(sql)
        len = imageLen(np.array(cur.fetchall()))

        sql = "select coordinate[0],coordinate[1] from vas_sample where id between " + id[0] + " and " + id[1]
        cur.execute(sql)
        vas_len = imageLen(np.array(cur.fetchall()))

        startInterval = IntervalChk(5000, 10, int(id[0]))
        endInterval = IntervalChk(5000, 10, int(id[1]))
        sql = "select count(*) from (select distinct x,y from ev where viz>" + str(
            startInterval[0]) + " and viz<" + str(endInterval[0]) + ") t"
        cur.execute(sql)
        mvs_len = cur.fetchall()[0][0]

        print float(vas_len) / len, float(mvs_len) / len, startInterval[0], endInterval[0]


def dp_upper_bound(v_alpha, v_theta, n):
    # np.save("alpha.json",v_alpha)
    # np.save("theta.json",v_theta)
    # v_alpha=np.load("alpha.npy")
    # v_theta=np.load("theta.npy")
    n_v_alpha = v_alpha / v_alpha.sum()

    # SUM[(theta[i]-alpha[i])^2]
    sum_square_difference = (np.square(v_theta - v_alpha)).sum()

    # SUM[(theta[i]+alpha[i])^2]
    sum_square_addition = (np.square(v_theta + v_alpha)).sum()

    # SUM(alpha[i])
    sum_alpha = v_alpha.sum()

    # SUM(theta[i])
    sum_theta = v_theta.sum()

    # SUM[(theta[i]+alpha[i])*(alpha[i]/SUM(alpha[i]))]
    sum_addition_theta_alpha_times_n_alpha = (np.multiply((v_alpha + v_theta), n_v_alpha)).sum()

    # SUM[(alpha[i]/SUM(alpha[i]))^2]
    sum_square_n_alpha = (np.square(n_v_alpha)).sum()

    a1 = (n * sum_square_difference - np.square(sum_alpha + sum_theta)) / (4.0 * n)
    b1 = (sum_alpha + sum_theta) / n
    c1 = - 1.0 / n

    a2 = (n * sum_square_addition - np.square(sum_alpha + sum_theta)) / (4.0 * n)
    b2 = (sum_alpha + sum_theta - n * sum_addition_theta_alpha_times_n_alpha) / n
    c2 = sum_square_n_alpha - (1.0 / n)

    # 4th order polynomial cooefficients
    p = [4.0 * ((a1 ** 2) * a2 - a1 * (a2 ** 2)),
         4.0 * ((a1 ** 2) * b2 + a1 * a2 * b1 - (a2 ** 2) * b1 - a1 * a2 * b2),
         4.0 * (a1 ** 2) * c2 + 4 * a1 * b1 * b2 + a2 * (b1 ** 2) - 4 * (a2 ** 2) * c1 - 4 * a2 * b1 * b2 - a1 * (
                 b2 ** 2),
         4.0 * a1 * b1 * c2 + (b1 ** 2) * b2 - 4.0 * a2 * b2 * c1 - b1 * (b2 ** 2),
         (b1 ** 2) * c2 - (b2 ** 2) * c1]

    # Solve this 4th order polynomial equation
    rs = np.roots(p)

    # for each r in rs, calculate the boundary
    maxDP1 = 0.0
    for i in range(0, len(rs)):
        r = rs[i]
        fr = np.sqrt(a1 * (r ** 2) + b1 * r + c1) + np.sqrt(a2 * (r ** 2) + b2 * r + c2)
        if r <= 0 or r > 1:
            continue
        if np.iscomplex(r):
            continue
        # get rid of false roots:
        left = 2.0 * a1 * r + b1
        right = 2.0 * a2 * r + b2
        if left * right > 0:
            continue
        if fr > maxDP1:
            maxDP1 = fr

    return maxDP1


def dp_time(startID, endID):
    s = time.time()
    coord1 = np.array(sample_seek(startID, endID))
    e = time.time()

    sql = "select coordinate[0],coordinate[1] from Bigtweets where ID between " + str(startID) + " and " + str(endID)
    ss = time.time()
    cur.execute(sql)
    coord2 = cur.fetchall()
    ee = time.time()

    sql = "select x,y from s_tweets1 where ID between " + str(startID) + " and " + str(endID)
    sss = time.time()
    cur.execute(sql)
    coord3 = cur.fetchall()
    eee = time.time()

    print e - s, len(coord1), eee - sss, len(coord3), ee - ss, len(coord2)


def dp_comparison(startID, endID):
    coord1 = np.array(sample_seek(startID, endID))
    ss_matrix = hashByNumpy(coord1, map, hv)

    sql = "select coordinate[0],coordinate[1] from Bigtweets where ID between " + str(startID) + " and " + str(endID)
    cur.execute(sql)
    original_matrix = hashByNumpy(np.array(cur.fetchall()), map, hv)

    startHV = IntervalChk(500, 10, startID)[0]
    endHV = IntervalChk(500, 10, endID)[0]

    sql = "select x,y,sum(density) from HV where viz >" + str(startHV) + " and viz<" + str(endHV) + " group by x,y"
    cur.execute(sql)
    mvs_inner_list = np.array(cur.fetchall())
    alpha_matrix = np.zeros(hv)
    for r in mvs_inner_list:
        alpha_matrix[r[0]][r[1]] = r[2]

    sql = "select x,y,sum(density) from HV where viz >=" + str(startHV) + " and viz<=" + str(endHV) + " group by x,y"
    cur.execute(sql)
    mvs_outer_list = np.array(cur.fetchall())
    theta_matrix = np.zeros(hv)
    for r in mvs_outer_list:
        theta_matrix[r[0]][r[1]] = r[2]

    ss_sum = float(np.sum(ss_matrix))
    original_sum = float(np.sum(original_matrix))
    alpha_sum = float(np.sum(alpha_matrix))

    ss_eps = 0.0
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ss_eps += (ss_matrix[i][j] / ss_sum - original_matrix[i][j] / original_sum) ** 2
    ss_eps = math.sqrt(ss_eps)

    alpha_original_eps = 0.0
    for i in range(0, len(alpha_matrix)):
        for j in range(0, len(alpha_matrix[0])):
            alpha_original_eps += (alpha_matrix[i][j] / alpha_sum - original_matrix[i][j] / original_sum) ** 2
    alpha_original_eps = math.sqrt(alpha_original_eps)

    alpha_theta_eps = dp_upper_bound(alpha_matrix.flatten(), theta_matrix.flatten(), 10000)

    # print "sample seek precision:", ss_eps, "mvs real precision:", alpha_original_eps,"mvs est precision:", alpha_theta_eps
    print ss_eps, alpha_original_eps, alpha_theta_eps


def dp_comparison_step(startID, endID):
    coord1 = np.array(sample_seek(startID, endID))
    ss_matrix = hashByNumpy(coord1, map, hv)

    ssfile = open("ssHeatmap.txt", "ab+")
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ssfile.write(str(i) + " " + str(j) + " " + str(float(ss_matrix[i][j]) / np.sum(ss_matrix)) + "\n")

    sql = "select coordinate[0],coordinate[1] from Bigtweets where ID between " + str(startID) + " and " + str(endID)
    cur.execute(sql)
    original_matrix = hashByNumpy(np.array(cur.fetchall()), map, hv)

    ssfile = open("orHeatmap.txt", "ab+")
    for i in range(0, len(original_matrix)):
        for j in range(0, len(original_matrix[0])):
            ssfile.write(
                str(i) + " " + str(j) + " " + str(float(original_matrix[i][j]) / np.sum(original_matrix)) + "\n")

    startHV = IntervalChk(500, 10, startID)[0]
    endHV = IntervalChk(500, 10, endID)[0]

    sql = "select x,y,sum(density) from HV where viz >=" + str(startHV) + " and viz<=" + str(endHV) + " group by x,y"
    cur.execute(sql)
    mvs_outer_list = np.array(cur.fetchall())
    theta_matrix = np.zeros(hv)
    for r in mvs_outer_list:
        theta_matrix[r[0]][r[1]] = r[2]

    ss_sum = float(np.sum(ss_matrix))
    original_sum = float(np.sum(original_matrix))

    ss_eps = 0.0
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ss_eps += (ss_matrix[i][j] / ss_sum - original_matrix[i][j] / original_sum) ** 2
    ss_eps = math.sqrt(ss_eps)

    for v in range(2, endHV + 1):
        sql = "select x,y,sum(density) from HV where viz >" + str(startHV) + " and viz<" + str(v) + " group by x,y"
        cur.execute(sql)
        mvs_inner_list = np.array(cur.fetchall())
        alpha_matrix = np.zeros(hv)
        for r in mvs_inner_list:
            alpha_matrix[r[0]][r[1]] = r[2]

        alpha_sum = float(np.sum(alpha_matrix))

        alpha_original_eps = 0.0
        for i in range(0, len(alpha_matrix)):
            for j in range(0, len(alpha_matrix[0])):
                alpha_original_eps += (alpha_matrix[i][j] / alpha_sum - original_matrix[i][j] / original_sum) ** 2
        alpha_original_eps = math.sqrt(alpha_original_eps)

        alpha_theta_eps = dp_upper_bound(alpha_matrix.flatten(), theta_matrix.flatten(), 10000)

        # print "sample seek precision:", ss_eps, "mvs real precision:", alpha_original_eps,"mvs est precision:", alpha_theta_eps
        print ss_eps, alpha_original_eps, alpha_theta_eps
