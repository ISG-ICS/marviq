import LIMITlib as lmt
import numpy as np
import vas

def histogram(nEV, nInterval):
    #Clear histogram
    lmt.clear_histogram("splitting_point")
    lmt.clear_histogram("parent_interval")
    lmt.clear_histogram("parent_pixels")
    lmt.clear_histogram("child_interval")
    lmt.clear_histogram("EV")
    lmt.clear_histogram("RV")
    print "Clear histogram, DONE."
    lmt.dividing_points(1000,11)
    #Create Histogram
    lmt.create_histogram(500,"bigtweets")
    #EV-Only
    # lmt.SnapShot(dt="BEV",nEV=140,nBucket=0,tab="taxi")
    #EV+RV
    lmt.SnapShot(dt="BEV",rdt="RV",nEV=nEV,nBucket=0,nInterval=nInterval, tab="bigtweets")
    print "Construct histogram, DONE."

def workload():
    #Create workload
    lmt.RandomQueries(nInterval=500,fn="random_ranges.txt")
    lmt.GetIntervals(nInterval=475,fn="random_ranges.txt",dstFn="random_workload_mvs+16.txt")
    lmt.GetIntervals(nInterval=485, fn="random_ranges.txt", dstFn="random_workload_mvs+8.txt")
    lmt.GetIntervals(nInterval=490, fn="random_ranges.txt", dstFn="random_workload_mvs+1.txt")
    lmt.GetIntervals(nInterval=500, fn="random_ranges.txt", dstFn="random_workload_mvs+32.txt")


def performance():
    #Compare performance of histogram
    lmt.ResultSize()
    lmt.QualityDiff()

#######################################  heatmap  ##############################################
# create the heatmap histogram, 10 EVs, 500 intervals.
def heatmapCmp():
    lmt.clear_histogram("HV")
    lmt.heatmap_histogram("HV",10,500,"bigtweets")

    lmt.RandomQueries(nInterval=500,fn="random_ranges.txt")
    f=open("random_ranges.txt")
    for line in f.readlines():
        ids=line.split()
        lmt.dp_time(int(ids[0]), int(ids[1]))

    lmt.dp_comparison_step(862753692393646275, 863340209606019381)
    lmt.dp_time(862753692393646275, 862760692393646275)

#######################################   VAS comparison   ###################################################
def compareVAS():
    lmt.sample_vas(3, 500)
    histogram(10, 2000)
    lmt.RandomQuereisForVAS(5000,"random_ranges_vas.txt")
    lmt.vas_quality("random_ranges_vas.txt")
    lmt.test(3)#100% range covered
    lmt.three_qualities(tau=0.948, fn='random_ranges_vas.txt',resFile="quality_random_ranges_vas.txt")

#######################################  create VAS samples   ###################################################
def vas_sample():
    epsilon = 0.00002#0.06299648839902239#vas.get_epsilon(pointarray)
    print "esp=", epsilon

    for sql in range(0, 4):
        lmt.cur.execute(sqllist[sql])
        pointarray = np.array(lmt.cur.fetchall())
        for i in range(1, 10):
            print tablist[sql], i
            pa = pointarray.copy()
            sample = vas.sample(pa, i*5000, epsilon)
            for r in sample:
                lmt.cur.execute("insert into vas_"+tablist[sql]+"_sample"+str(i)+" values (random(), point("+str(r[0])+","+str(r[1])+"))")
            lmt.cur.execute("commit")

# New York
nysql="select coordinate[0] as p0, coordinate[1] as p1 from tweets where coordinate<@box(point(-76.957011,41.937384),point(-71.200176,39.204970)) and id > 820000009277976580 order by id limit 200000"
#US
ussql="select coordinate[0] as p0, coordinate[1] as p1 from tweets where id > 820000009277976580 order by id limit 200000"
#Chicago
chsql="select coordinate[0] as p0, coordinate[1] as p1 from tweets where coordinate<@box(point(-91.653717,45.404041),point(-79.164748,38.165591)) and id > 820000009277976580 order by id limit 200000"
#LA
lasql="select coordinate[0] as p0, coordinate[1] as p1 from tweets where coordinate<@box(point(-118.660396,34.328978),point(-117.464574,33.448546)) and id > 820000009277976580 order by id limit 200000"

sqllist=[nysql, ussql, chsql, lasql]
tablist=['ny', 'us', 'ch', 'la']



lmt.cur.execute(nysql)
pa=np.array(lmt.cur.fetchall())
print "original:", lmt.imageLen(pa)

lmt.cur.execute("select p[0] as p0, p[1] as p1 from vas_ny_sample5")
pa=np.array(lmt.cur.fetchall())
print "approximate:", lmt.imageLen(pa)
