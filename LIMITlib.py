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
map=us_map

rv=(32,18)
ev=(480,270)
hv=(100,100)

yStep=(y1-y0)/res_y
xStep=(x1-x0)/res_x

##############Tweets ID########################################################
maxID=1009129304360288256
minID=862500001267011588
interval_size=293258606186553 #500
startID=862500001267011588
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

#postgresql connection
conStr = "dbname='postgres' user='postgres' host='192.168.209.1' port='5432' password='liming' "
conn = psycopg2.connect(conStr)
cur = conn.cursor()

#oracle connection
# ora_conn=cx_Oracle.connect("system","Oracle123","curium.ics.uci.edu:1521/orcl")
# ora_cur=ora_conn.cursor()

def CreateBiasedDS(scale,newTab,srcTab='tweets',col='coordinate'):
    try:
        #cur.execute("truncate "+newTab)
        print 'Old table found, deleted'
    except psycopg2.InternalError:
        print 'No old table found, creating table.'
    cur.execute("create table "+newTab+" as select * from "+srcTab+" where 1=2")
    xlist=range(0,scale)
    ylist=range(0,scale)
    random.shuffle(xlist)
    random.shuffle(ylist)
    xstep=(x1-x0)/scale
    ystep=(y1-y0)/scale
    for x in xlist:
        for y in ylist:
            rect="box'(("+str(x0+x*xstep)+","+str(y0+y*ystep)+"),("+str(x0+(x+1)*xstep)+","+str(y0+(y+1)*ystep)+"))'"
            sql="insert into "+newTab+" select * from "+srcTab+" where "+col+" <@"+rect+" order by "+col+"<->point(0,0)"
            cur.execute(sql)
    cur.execute("commit")
    print "Done."

def restart(version=9.6):
    if sys.platform == 'darwin':
        os.system('brew services stop postgresql')
        os.system('brew services start postgresql')
    elif sys.platform == 'linux2':
        if version >= 9.5:
            print 'sudo systemctl restart postgresql-' + str(version)
            os.system('sudo systemctl restart postgresql-' + str(version))
        else:
            os.system('sudo systemctl restart postgresql')

    i = 0
    while i <= 10:
        try:
            conn = psycopg2.connect(conStr)
            cur = conn.cursor()
            break
        except psycopg2.DatabaseError:
            print 'wait 1s for db restarting ... ...'
            time.sleep(1)
            i += 1
    if i > 10:
        raise psycopg2.DatabaseError


# Return the coordinate of keyword from table tb, if limit is -1, then return all the records, order by is the id of the table.
def GetCoordinate(tb, keyword, limit=-1, orderby=False):
    conn = psycopg2.connect(conStr)
    cur = conn.cursor()
    sql = "select count(*) from information_schema.columns where table_name='"+tb+"' and column_name='coordinate'"
    cur.execute(sql)
    hasPoint=cur.fetchall()[0][0]
    if int(hasPoint) == 1:
        sql = " select coordinate[0],coordinate[1] from " + tb + " where to_tsvector('english',text)@@to_tsquery('english','" + keyword + "')"
    else:
        sql = "select x,y from " + tb + " where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"')"
    if orderby:
        sql += " order by id"
    if limit >= 0:
        sql += " limit " + str(limit)
    cur.execute(sql)
    return cur.fetchall()
# get coordinates from oracle
def GetCoordinateOra(tb, keyword, limit=-1, orderby=False):
    sql = "select x,y from " + tb + " where contains(text,'"+keyword+"')>0"
    if orderby:
        sql += " order by id"
    if limit >= 0:
        sql += " where rownum>= " + str(limit)
    ora_cur.execute(sql)
    return ora_cur.fetchall()
def GetCoordinateSMP(tb,keyword,r,method):
    sql=""
    if method=='B':
        sql="select coordinate[0],coordinate[1] from "+tb+" tablesample bernoulli("+str(r)+") where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"')"
    else:
        sql="select coordinate[0],coordinate[1] from "+tb+" tablesample system("+str(r)+") where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"')"
    cur.execute(sql)
    return cur.fetchall()
def GetCoordinateUber(tb, base, limit=-1, orderby=False):
    conn = psycopg2.connect(conStr)
    cur = conn.cursor()
    sql = "select coordinate[0],coordinate[1] from " + tb #+ " where category='"+base+"'"
    if limit >= 0:
        sql += " limit " + str(limit)
    cur.execute(sql)
    return cur.fetchall()

def GetCoordinateRange(tb,startID, endID):
    conn = psycopg2.connect(conStr)
    cur = conn.cursor()
    sql = "select coordinate[0],coordinate[1] from " + tb +" where id between "+str(startID)+" and "+str(endID)
    cur.execute(sql)
    return cur.fetchall()

# Return the keywords in table tb, the lower and upper are the frequency bounds, k is the limit number of returned keywords.
def GetKeywords(tb, lower, upper, k):
    conn = psycopg2.connect(conStr)
    cur = conn.cursor()
    sql = "select vector,count from " + tb + " where count>=" + str(lower) + " and count<" + str(
        upper) + "order by count limit " + str(
        k)  # +" and vector not in (select distinct keyword from keyword_k_q) order by count"
    cur.execute(sql)
    return cur.fetchall()
# Map the coodrinates into cells, the type of 'ar' is the numpy array, r is the coordinate range of the map. the returned value H is the matrix of cells,
# each value is the number of records in the cell.
def  hashByNumpy(ar, r=map,b=(res_x, res_y)):
    H, x, y = np.histogram2d(ar[:, 0], ar[:, 1], bins=b, range=r)
    return H

# L:LIMIT, S: Block based table sample, B: Record based table sample
def TimeQuality(wList,TabList,method='L',RatioL=1,RatioH=90,kStep=5):
    for w in wList:
        for tab in TabList:
            fullRec=GetCoordinate(tab,w)
            pImageLen=imageLen(np.array(fullRec))
            for r in range(RatioL,RatioH+1,kStep):
                if method=='L':
                    k=int(float(r)/100.0*len(fullRec))
                    sql="select coordinate from "+tab+" where to_tsvector('english',text)@@to_tsquery('english','"+w+"') limit "+str(k)
                    print w,tab,'LIMIT',len(fullRec),r,imageLen(np.array(fullRec[:k]))/float(pImageLen),SQLexeTime(sql)
                elif method=='B':
                    sql="select coordinate from "+tab+" tablesample bernoulli("+str(r)+") where to_tsvector('english',text)@@to_tsquery('english','"+w+"')"
                    print w,tab,'BERNOULLI',len(fullRec),r,imageLen(np.array(GetCoordinateSMP(tab,w,r,method)))/float(pImageLen),SQLexeTime(sql)
                elif method=='S':
                    sql="select coordinate from "+tab+" tablesample system("+str(r)+") where to_tsvector('english',text)@@to_tsquery('english','"+w+"')"
                    print w,tab,'SYSTEM',len(fullRec),r,imageLen(np.array(GetCoordinateSMP(tab,w,r,method)))/float(pImageLen),SQLexeTime(sql)
                else:
                    print 'Unrecognized sampling method:', method


def imageLen(array):
    return np.count_nonzero(hashByNumpy(array))

# return the mse of two matrix
def myMSE(m1,m2, binary=True):#m1, m2 are the matrixs of the ground-truth map and approximate map
    if binary:
        m1=np.where(m1>0,1,0) #convert each element to 0 or 1
        m2=np.where(m2>0,1,0) #convert each element to 0 or 1
    err=0
    for i in range(0, len(m1)):
        for j in range(0,len(m1[0])):
            err+=(m1[i][j]-m2[i][j])**2
    return math.sqrt(err)/(len(m1)*len(m1[0]))
    # err=math.sqrt(np.sum((m1-m2)**2))
    # err/=float(len(m1)*len(m1[0]))
    # return err
#Use binary search to find the k that have quality Q in coordinate ar.
def findkofQ(ar, Q):
    perfectLen = imageLen(np.array(ar))
    i = 0.0
    l = 0.0
    h = 100.0
    similarity = 0.0
    iterTimes = 0
    while (similarity < 0.85 or similarity > 0.86) and iterTimes < 10:
        if similarity < 0.85:
            l = i
            i = (h + i) / 2
        else:
            h = i
            i = (i + l) / 2
        k = int(i * len(ar) / 100)
        sampleLen = imageLen(np.array(ar[:k]))
        similarity = float(sampleLen) / perfectLen
        iterTimes += 1
    return i

#Find the k of hybrid queries, w:keyword, q:quality, tb: original data table, hybridtab: offline sample table
def kOfHybridQueries(w, q, tb,hybridtab='null'):
    coord = GetCoordinate(tb, w, -1)
    if len(coord) < 5000:
        return 0
    offlineHs = np.zeros(shape=ev, dtype=int)
    if hybridtab is not 'null':
        offlinecoord = GetCoordinate(hybridtab, w, -1)
        offlineHs = hashByNumpy(np.array(offlinecoord))
    ar = np.array(coord)
    H = hashByNumpy(ar)#matrix of from the original data table
    perfectLen = np.count_nonzero(H)
    i = 0.0
    l = 0.0
    h = 100.0
    similarity = 0.0
    iterTimes = 0
    #binary search of k for quality q, max iteration times is 20
    while (similarity < q or similarity > q * 1.01) and iterTimes < 20:
        if similarity < q:
            l = i
            i = (h + i) / 2
        else:
            h = i
            i = (i + l) / 2
        k = int(i * len(ar) / 100)
        Hs = hashByNumpy(ar[:k])
        if hybridtab is not 'null':#combine the online subset with the offline subset
            Hs += offlineHs
        sampleLen = np.count_nonzero(Hs)
        similarity = float(sampleLen) / perfectLen
        iterTimes += 1
    return k
def kOfHybridQueriesUber(w, q, tb,hybridtab='null'):
    coord = GetCoordinateUber(tb, w, -1)
    if len(coord) < 5000:
        return 0
    offlineHs = np.zeros(shape=ev, dtype=int)
    if hybridtab is not 'null':
        offlinecoord = GetCoordinateUber(hybridtab, w, -1)
        offlineHs = hashByNumpy(np.array(offlinecoord))
    ar = np.array(coord)
    H = hashByNumpy(ar)#matrix of from the original data table
    perfectLen = np.count_nonzero(H)
    i = 0.0
    l = 0.0
    h = 100.0
    similarity = 0.0
    iterTimes = 0
    #binary search of k for quality q, max iteration times is 20
    while (similarity < q or similarity > q * 1.01) and iterTimes < 20:
        if similarity < q:
            l = i
            i = (h + i) / 2
        else:
            h = i
            i = (i + l) / 2
        k = int(i * len(ar) / 100)
        Hs = hashByNumpy(ar[:k])
        if hybridtab is not 'null':#combine the online subset with the offline subset
            Hs += offlineHs
        sampleLen = np.count_nonzero(Hs)
        similarity = float(sampleLen) / perfectLen
        iterTimes += 1
    return k
# find k of hybird queries in oracle, the only difference is call of GetCoordinate
def kOfHybridQueriesOra(w, q, tb,hybridtab='null'):
    coord = GetCoordinateOra(tb, w, -1)
    if len(coord) < 5000:
        return 0
    offlineHs = np.zeros(shape=ev, dtype=int)
    if hybridtab is not 'null':
        offlinecoord = GetCoordinateOra(hybridtab, w, -1)
        offlineHs = hashByNumpy(np.array(offlinecoord))
    ar = np.array(coord)
    H = hashByNumpy(ar)#matrix of from the original data table
    perfectLen = np.count_nonzero(H)
    i = 0.0
    l = 0.0
    h = 100.0
    similarity = 0.0
    iterTimes = 0
    #binary search of k for quality q, max iteration times is 20
    while (similarity < q or similarity > q * 1.01) and iterTimes < 20:
        if similarity < q:
            l = i
            i = (h + i) / 2
        else:
            h = i
            i = (i + l) / 2
        k = int(i * len(ar) / 100)
        Hs = hashByNumpy(ar[:k])
        if hybridtab is not 'null':#combine the online subset with the offline subset
            Hs += offlineHs
        sampleLen = np.count_nonzero(Hs)
        similarity = float(sampleLen) / perfectLen
        iterTimes += 1
    return k
#fine the trend of k when scaling ataset
def ScaleDataSize(kwList,tab):
    for w in kwList:
        coord = GetCoordinate(tab, w, -1)
        for s in range(10, 101, 10):
            size = s * len(coord) / 100
            scoord = coord[:size]
            r = findkofQ(scoord, 0.85)
            print w, 'dataset size:',size,'85% quality:',r * size/100

#Load state polygons to db from file
def loadStatePolygon():
    poly = demjson.decode_file("state.json")
    for state in poly['features']:
        name = state['properties']['name']
        polys = state['geometry']['coordinates']
        for p in polys:
            coords = "'" + str(p).replace('[', '(').replace(']', ')')[1:-1] + "'"
            sql = "insert into statepolygon values('" + name + "'," + coords + ")"
            cur.execute(sql)
            print name
    cur.execute('commit')

#update the column of state in coordtweets
def updateStateField():
    cur.execute("select id from coordtweets")
    ids = cur.fetchall()
    i = 0
    for id in ids:
        name = "NULL"
        cur.execute("select state from statepolygon where poly@>(select coordinate from coordtweets where id=" + str(
            id[0]) + ") limit 1")
        res = cur.fetchall()
        if len(res) > 0:
            name = res[0][0]
        cur.execute("update coordtweets set state='" + name + "' where id=" + str(id[0]))
        i += 1
        print i, id
    cur.execute('commit')

#produce count map of subset LIMIT k
def countMap(w, k=4000000):
    sql = "select state, count(*) from (select state,id from coordtweets where to_tsvector('english',text)@@to_tsquery('english','" + w + "') limit " + str(
        k) + ") t group by t.state"
    cur.execute(sql)
    return cur.fetchall()

#use the distributed precision to compute the count map quality, s and e are the start and end frequency
def countMapQuality(s, e):
    keywords = GetKeywords('vectorcount', s, e, 1000)
    for w in keywords:
        gt = dict((x, y) for x, y in countMap(w[0]))
        for i in gt.keys():
            gt[i] = float(gt[i]) / w[1]
        for r in range(1, 101, 1):
            k = r * w[1] / 100
            sub = dict((x, y) for x, y in countMap(w[0], k))
            for i in sub.keys():
                sub[i] = float(sub[i]) / k
            e = 0.0
            for i in gt.keys():
                if sub.has_key(i):
                    e += math.pow((gt[i] - sub[i]), 2)
                else:
                    e += math.pow(gt[i], 2)
            print w[0], r / 1000, k, math.sqrt(e)


def getError(gt, freq, sub, k):
    e = 0.0
    for i in gt.keys():
        if sub.has_key(i):
            e += math.pow(float(gt[i]) / freq - float(sub[i]) / k, 2)
        else:
            e += math.pow(float(gt[i]) / freq, 2)
    return math.sqrt(e)


def countMapQualityMem(s, e):
    keywords = GetKeywords('vectorcount', s, e, 1000)
    for w in keywords:
        cur.execute(
            "select state from coordtweets where to_tsvector('english',text)@@to_tsquery('english','" + w[0] + "')")
        res = cur.fetchall()
        freq = len(res)
        gt = {}
        for i in res:
            if gt.has_key(i[0]):
                gt[i[0]] += 1
            else:
                gt[i[0]] = 1
    for r in range(1, 201, 1):
        k = r * w[1] / 1000
        sub = {}
        for i in range(0, k):
            if sub.has_key(res[i][0]):
                sub[res[i][0]] += 1
            else:
                sub[res[i][0]] = 1
        print w[0], float(k) / w[1], getError(gt, freq, sub, k)

#compare k in online, online+offline, s,e are the start and end frequencies, tb is the original data table
def kComparison(s, e, tb):
    keywords = GetKeywords('vectorcount', s, e, 100)
    for w in keywords:
        online = kOfHybridQueries(w[0], 0.85, tb)#online
        offset0 = (kOfHybridQueries(w[0], 0.85, tb, 'gridsample0'))#online+stratified sample
        offset50 = (kOfHybridQueries(w[0], 0.85, tb, 'gridsample50'))#onlien+stratified sample+ sample from tail
        offsetalpha = (kOfHybridQueries(w[0], 0.85, tb, 'gridsample'))#onlien+stratified sample+ sample from tail+reducing #records in cells of LIMIT k
        if online > 0:
            print w[0], w[1], online, offset0, offset50, offsetalpha
# find the k of each cell that how many records need to be scanned to find the keyword
def FindFirstIndexofKeyword(keyword):
    for x in range(0,res_x):
        for y in range(0,res_y):
            bottomleftX=x0+xStep*x
            bottomleftY=y0+yStep*y
            toprightX=x0+xStep*(x+1)
            toprightY=y0+yStep*(y+1)
            box="box '("+str(bottomleftX)+","+str(bottomleftY)+"),("+str(toprightX)+","+str(toprightY)+")'"
            sql="select text from coordtweets where "+box+"@>coordinate"
            cur.execute(sql)
            texts=cur.fetchall()
            i=0
            found=False
            for text in texts:
                i+=1
                sql="select to_tsvector('english','"+text[0].replace("'"," ")+"')@@to_tsquery('english','"+keyword+"')"
                cur.execute(sql)
                result=cur.fetchall()
                if result[0][0]:
                    found=True
                    break
            if found:
                cur.execute("insert into firstindex values('"+keyword+"',"+str(x)+","+str(y)+","+str(i)+","+str(len(texts))+")")
            else:
                cur.execute("insert into firstindex values('"+keyword+"',"+str(x)+","+str(y)+","+str(0)+","+str(len(texts))+")")
            cur.execute("commit")
#find the max density of the map
def maxDensity(tb):
    for x in range(0,res_x):
        for y in range(0,res_y):
            bottomleftX=x0+xStep*x
            bottomleftY=y0+yStep*y
            toprightX=x0+xStep*(x+1)
            toprightY=y0+yStep*(y+1)
            box="box '("+str(bottomleftX)+","+str(bottomleftY)+"),("+str(toprightX)+","+str(toprightY)+")'"
            sqlcnt="select count(*) from "+tb+" where "+box+"@>coordinate"
            cur.execute(sqlcnt)
            cnt=cur.fetchall()
            if cnt[0][0]>dmax:
                dmax=cnt[0][0]

#alpha=0: use pure stratified sampling
#alpha=x: the #records in each cell is proportional to its density, the cells that density>(1/alpha)* max_density have no records.
def stratifiedSampling(k,alpha=0):
    i=0
    j=0
    dmax=maxDensity('coordtweets')#the max density of coordtweets is 399,000.
    for x in range(0,res_x):
        for y in range(0,res_y):
            tmpoffset=0
            bottomleftX=x0+xStep*x
            bottomleftY=y0+yStep*y
            toprightX=x0+xStep*(x+1)
            toprightY=y0+yStep*(y+1)
            box="box '("+str(bottomleftX)+","+str(bottomleftY)+"),("+str(toprightX)+","+str(toprightY)+")'"
            sqlcnt="select count(*) from coordtweets where "+box+"@>coordinate"
            cur.execute(sqlcnt)
            cnt=cur.fetchall()
            if cnt[0][0]==0:
                continue
            tmpk=int(k*float(max(0,dmax-alpha*cnt[0][0]))/dmax)
            if cnt[0][0]>=tmpk:
                tmpoffset=cnt[0][0]-tmpk
            else:
                tmpoffset=0
            sql="insert into gridsample select * from coordtweets where "+box+"@>coordinate offset "+str(tmpoffset)+" limit "+str(tmpk)
            cur.execute(sql)
            print res_x,x,res_y,y,cnt[0][0],tmpoffset,tmpk
    cur.execute('commit')
    print "Grid Sample: k="+str(k)
# Get curves of keyword w in table tab, start k=10%, end k=90%
def Curves(w,tab):
    coord=GetCoordinate(tab,w)
    print w,len(coord)
    perfectImageLen=imageLen(np.array(coord))
    for r in range(10,100,10):
        subLen=int(float(r)*len(coord)/100.0)
        aprxImageLen=imageLen(np.array(coord[:subLen]))
        print r,float(aprxImageLen)/perfectImageLen

#k: the threshold of #records for each cell
#refTab: the table created by using LIMIT k of original datatable without contains keyword.
def gridSampleTopCells(k,refTab,smpTab,srcTab):
    cur.execute("create table if not exists "+smpTab+" as select * from tweets where 1=2")
    cur.execute("commit")
    totaltime=0
    for x in range(0,res_x):
        for y in range(0,res_y):
            tmpoffset=0
            bottomleftX=x0+xStep*x
            bottomleftY=y0+yStep*y
            toprightX=x0+xStep*(x+1)
            toprightY=y0+yStep*(y+1)
            box="box '("+str(bottomleftX)+","+str(bottomleftY)+"),("+str(toprightX)+","+str(toprightY)+")'"
            #remove top n cells
            sql="select count(*) from "+refTab+" where "+box+"@>coordinate"
            cur.execute(sql)
            cnt=cur.fetchall()[0][0]
            if cnt>=k:
                continue
            else:
                tmpk=k-cnt
            # sqlcnt="select count(*) from tweets where "+box+"@>coordinate"
            # cur.execute(sqlcnt)
            # cnt=cur.fetchall()
            # if cnt[0][0]>=tmpk:
            #     tmpoffset=cnt[0][0]-tmpk
            # else:
            #     tmpoffset=0
            # if cnt[0][0]>0:
            t1=time.time()
            sql="insert into "+smpTab+" select * from "+srcTab+" where "+box+"@>coordinate offset "+str(cnt)+" limit "+str(tmpk)##str(tmpoffset)
            cur.execute(sql)
            t2=time.time()
            print res_x,x,res_y,y,cnt,tmpk
            totaltime+=t2-t1
    cur.execute('commit')
    print "Grid Sample: k="+str(k)+", net time:"+str(totaltime)

# using the random function to get a random sample for each cell.
def gridSampleRandomFunction():
    cur.execute("create table if not exists vas_ss3 as select * from tweets where 1=2")
    cur.execute("commit")
    totaltime=0
    for x in range(0,res_x):
        for y in range(0,res_y):
            tmpoffset=0
            bottomleftX=x0+xStep*x
            bottomleftY=y0+yStep*y
            toprightX=x0+xStep*(x+1)
            toprightY=y0+yStep*(y+1)
            box="box '("+str(bottomleftX)+","+str(bottomleftY)+"),("+str(toprightX)+","+str(toprightY)+")'"
            #remove top n cells
            sql="select count(*) from ss3 where "+box+"@>coordinate"
            cur.execute(sql)
            cnt=cur.fetchall()[0][0]
            if cnt==0:
                continue
            print x,y
            sql="select count(*) from rnd5 where "+box+"@>coordinate"
            cur.execute(sql)
            cnt2=cur.fetchall()[0][0]
            r=float(cnt)/float(cnt2)
            sql="insert into vas_ss3 select * from rnd5 where "+box+"@>coordinate and random()<="+str(r)
            cur.execute(sql)
    cur.execute('commit')
    print "Grid Sample: k="+str(k)+", net time:"+str(totaltime)
#get k of original, offline sample. tab is the original data table, ss is the sample lsit, wlist is the keyword list, quality is the specified quality
def KofQueries(tab,ss,wlist,quality):
    kwList=wlist##freq: 50k,500k,1M,2M
    stratSampleList=[ss]
    origTab=tab
    for kw in kwList:
        ##A. Original query, get the number of all records that contain the keyword, and time
        freq=len(lmt.GetCoordinate(origTab,kw,-1))
        print tab,kw,freq,'null','0','1'

        ##B. Online sampling (LIMIT K), get the number of records of quality=quality, and time
        for q in quality:
            k=lmt.kOfHybridQueries(kw,q,origTab)
            print tab,kw,k,'null','0',q

            ##C. Online sampling + Offline sampling
            for smp in stratSampleList:
                k0=len(lmt.GetCoordinate(smp, kw, -1))## #records in offline sample
                k1=lmt.kOfHybridQueries(kw,q,origTab,smp)
                print tab,kw,k1,smp,k0,q
def SQLexeTime(sql,times=1):
    totalTime=0.0
    for i in range(0,times):
        cur.execute("select count(*) from (select coordinate from dummyTab) a")
        cur.execute("select count(*) from dummyTab where to_tsvector('english',text)@@to_tsquery('english','veteran')")
        s=time.time()
        cur.execute(sql)
        e=time.time()
        totalTime+=e-s
    return totalTime/times
# get execution time of keyword on table. dataset is the original data table, k1 is the k of original data table, smpTab is the sample table, k0 is the k of sample table.
def getExeTime(dataset='BiasedUber',keyword='B02512',k1=1000,smpTab='smp',k0=100):
    #limitSQL="select * from "+dataset+" where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"') limit "+k1
    #sampleSQL="select * from "+smpTab+" where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"') limit "+k0
    limitSQL="select * from BiasedUber tablesample system(65) where category='B02512'"
    sampleSQL="select * from BiasedUber where category='B02512'"
    ###
    dummySQL="select count(*) from (select coordinate from dummytab) a"
    dummySQL2="select count(*) from tweets where to_tsvector('english',text)@@to_tsquery('english','veteran')"
    limitT=0.0
    sampleT=0.0
    for i in range(0,5):
        #lmt.restart()
        cur.execute(dummySQL)
        cur.execute(dummySQL2)
        ts=time.time()
        cur.execute(limitSQL)
        te=time.time()
        limitT+=te-ts
        cur.execute(dummySQL)
        cur.execute(dummySQL2)
        ts=time.time()
        cur.execute(sampleSQL)
        te=time.time()
        sampleT+=te-ts
    return limitT/5.0,sampleT/5.0
# get the accessed blocks of a query in postgresql.
def CountBlocks(dataset,keyword,k1,smpTab,k0):
    explainSQL="explain(analyze,buffers) "+"select * from "+dataset+" where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"') limit "+k1
    explainSQL2="explain(analyze,buffers) "+"select * from "+smpTab+" where to_tsvector('english',text)@@to_tsquery('english','"+keyword+"') limit "+k0
    lmt.cur.execute(explainSQL)
    lines=lmt.cur.fetchall()
    blocks2=""
    blocks1=lines[5]
    if smpTab!='null':
        lmt.cur.execute(explainSQL2)
        lines=lmt.cur.fetchall()
        blocks2=lines[5]
    print blocks1,blocks2
#find k of the category of biaseduber
def kInUber(category='B02512',tab='BiasedUber',quality=0.85):
    coord=GetCoordinateUber(tab,category)
    pLen=np.count_nonzero(hashByNumpy(np.array(coord),r=((-75, -72), (39, 43))))
    r=1
    sLen=0
    while(sLen<quality*pLen):
        r+=1
        k=int(r/100.0*len(coord))
        sLen=np.count_nonzero(hashByNumpy(np.array(coord[:k]),r=((-75, -72), (39, 43))))
    print r
# tweets id quality test
def idQuality(startID=820000001267011588, interval=1891293030932766,quality=0.85):
    sql="select coordinate[0],coordinate[1] from biasedtweets where id>="+str(startID)+" and id<"+str(startID+interval)
    cur.execute(sql)
    coord=cur.fetchall()
    pLen=imageLen(np.array(coord))
    q=0.0
    r=1
    while q<quality:
        r+=1
        k=int(r/100.0*len(coord))
        sLen=imageLen(np.array(coord[:k]))
        q=float(sLen)/pLen
        print pLen,sLen,r,q
def idQualityRnd(startID=820000001267011588, interval=1891293030932766,quality=0.85):
    sql="select coordinate[0],coordinate[1] from biasedtweets where id>="+str(startID)+" and id<"+str(startID+interval)
    cur.execute(sql)
    coord=cur.fetchall()
    pLen=imageLen(np.array(coord))
    q=0.0
    r=20
    while q<quality:
        r+=1
        sql="select coordinate[0],coordinate[1] from biasedtweets tablesample system("+str(r)+") where id>="+str(startID)+" and id<"+str(startID+interval)
        cur.execute(sql)
        scoord=cur.fetchall()
        sLen=imageLen(np.array(scoord))
        q=float(sLen)/pLen
        print pLen,sLen,r,q

# test of execution time
def exe():
    #dummySQL1="select count(*) from (select * from dummyTab) t"
    dummySQL2="select count(*)from (select * from dummyTab where id<1009129304360288256 and id>820000001267011588) t"

    for ss in range(0,3):
        for i in [3,5,9]:
            sql1="(select coordinate from biasedtweets where id>=820000001267011588 and id<821891294297944354 limit "+str(i/100.0*1264038)+") t"
            sql2="(select coordinate from biasedtweets tablesample system("+str(i*10)+") where id>=820000001267011588 and id<821891294297944354) t"
            sql3="(select coordinate from biasedtweets tablesample system("+str(i*10)+")) t"
            #sql="select count(*) from (select * from sBiasedUber tablesample system("+str((i+1)*10)+") where category='B02512') t"
            #sql="select count(*) from (select * from sBiasedUber where category='B02512' limit "+str(int((i+1)*0.1*205673)) + " ) t"
            #sql="select count(*) from (select * from sBiasedUber tablesample system("+str((i+1)*10)+") ) t"
            t=0
            for j in range(0,3):
                #cur.execute(dummySQL1)
                cur.execute(dummySQL2)
                if ss==0:
                    sql=sql1
                elif ss==1:
                    sql=sql2
                else:
                    sql=sql3
                t1=time.time()
                cur.execute(sql)
                t2=time.time()
                t=t+t2-t1
            print ss,i,t/3.0
#Find k and r of different ranges in Tweets.
def RangeQ(quality=0.85,tab='bigtweets',binSize=1000, ranges=[200,500]):
    maxID=1009129304360288256
    minID=820000001267011588
    step=(maxID-minID)/binSize
    startID=minID#860473672128972652
    endID=minID
    resultFile=open("RangeQ.txt",'w')
    resultFile.close()
    resultFile=open("RangeQ.txt",'a')
    for times in ranges:
        startID=minID
        while(startID<maxID):
            endID=startID+step*times
            sql="select coordinate[0],coordinate[1] from "+tab+" where id>="+str(startID)+" and id<"+str(endID)
            #print sql
            cur.execute(sql)
            coord=np.array(cur.fetchall())
            if len(coord>0):
                pLen=imageLen(coord)
                q=0.0
                r=10
                k=0
                while q<quality:
                    k=int(r/100.0*len(coord))
                    sLen=imageLen(coord[:k])
                    q=float(sLen)/pLen
                    r+=1
                print times,startID,endID,len(coord),r,k
                resultFile.writelines(str(times)+ " " +str(startID)+ " " +str(endID)+ " " +str(len(coord))+ " " +str(r)+ " " +str(k)+'\n')
                resultFile.flush()
            startID=endID
#Find #records and ratio in random ranges.
def RangeMerge(tab='biasedtweets'):
    maxID=1009129304360288256
    minID=862500001267011588
    step=(maxID-minID)/1000
    for bin in range(0,1):
        startID=minID+step*10*12*bin
        endID=startID+step*10*12
        #sql="select coordinate[0],coordinate[1] from "+tab+" where id>="+str(startID)+" and id<"+str(endID)
        sql="select coordinate[0],coordinate[1] from "+tab+" limit 5000000 offset "+str(bin*10000000+20000000)
        cur.execute(sql)
        coord=np.array(cur.fetchall())
        if len(coord>0):
            pLen=imageLen(coord)
            print 'Group','Ratio','#Record','#Point','#SubsetRecord','#SubsetPoint','Quality'+str(bin)
            for r in range(1,11):
                k=int(float(r)/10.0*len(coord))
                sLen=imageLen(coord[:k])
                print bin,r*10,len(coord),pLen,k,sLen,float(sLen)/pLen
# A, B and A&B subset relationship
def IsSubset():
    sqlA="select coordinate[0],coordinate[1] from bigtweets where id>820000001267011588 and id<=823000001267011588"
    sqlB="select coordinate[0],coordinate[1] from bigtweets where id>823000001267011588 and id<826000001267011588"
    sqlAB="select coordinate[0],coordinate[1] from bigtweets where id>820000001267011588 and id<=824000001267011588"
    cur.execute(sqlA)
    coordA=cur.fetchall()
    cur.execute(sqlB)
    coordB=cur.fetchall()
    cur.execute(sqlAB)
    coordAB=cur.fetchall()
    print 'A',imageLen(np.array(coordA))
    print 'B',imageLen(np.array(coordB))
    print 'AB-B',imageLen(np.array(coordAB[:len(coordA)]))
    print 'AB-A',imageLen(np.array(coordAB[len(coordA):]))
    print 'AB',imageLen(np.array(coordAB))
    print 'A+k',imageLen(np.array(coordAB[:len(coordA)+500000]))
    print 'k',imageLen(np.array(coordAB[len(coordA):len(coordA)+500000]))
#hashByNumpy(ar, r=((-170, -60), (15, 70)),b=(res_x, res_y)):


def clear_histogram(tab=""):
    sql="delete from "+tab
    cur.execute(sql)
    cur.execute("commit")

def dividing_points(nEV,point_num):
    id=minID
    intervalSize=(maxID-minID)/nEV
    for i in range(0, point_num):
        sql="insert into splitting_point values("+str(id)+",0,0,0)"
        cur.execute(sql)
        cur.execute("commit")
        id = id + intervalSize
def create_histogram(nEV,tab, nBucket=10,nInterval=10):
    sql="select point from splitting_point order by point asc"
    cur.execute(sql)
    points=cur.fetchall()
    startPoint=points[0]
    parent_id=0.0
    for endPoint in points:
        #retrieve data
        sql="select coordinate[0],coordinate[1] from "+tab+" where id between "+str(startPoint[0])+" and "+str(endPoint[0])
        cur.execute(sql)
        coord=np.array(cur.fetchall())
        if len(coord)<1:
            continue
        #a new parent intval
        sql="insert into parent_interval(parent_id,startval,endval) values("+str(parent_id)+","+str(startPoint[0])+","+str(endPoint[0])+")"
        cur.execute(sql)
        cur.execute("commit")
        #records for the new parent_interval
        OriginalViz=hashByNumpy(coord,r=map,b=ev)
        for x in range(0,ev[0]):
            for y in range(0,ev[1]):
                if OriginalViz[x][y]!=0:
                    sql="insert into parent_pixels(parent_id,x,y) values("+str(parent_id)+","+str(x)+","+str(y)+")"
                    cur.execute(sql)
        cur.execute("commit")
        #child intervals
        for r in range(0,nBucket):
            ks=int(r*10/100.0*len(coord))
            ke=int((r+1)*10/100.0*len(coord))
            tmpVizA=hashByNumpy(coord[0:ke],r=map,b=ev)
            LowVizA=hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizA))),r=((0,ev[0]),(0,ev[1])),b=rv)
            ###reversed value
            tmpVizB=hashByNumpy(coord[ks-len(coord):],r=map,b=ev)
            LowVizB=hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizB))),r=((0,ev[0]),(0,ev[1])),b=rv)

            for x in range(0,rv[0]):
                for y in range(0,rv[1]):
                    if LowVizA[x][y]!=0 or LowVizB[x][y]!=0:
                        sql="insert into child_interval(parent_id,child_id,x,y,a,b) values("+str(parent_id)+","+str(r)+","+str(x)+","+str(y)+","+str(LowVizA[x][y])+","+str(LowVizB[x][y])+")"
                        cur.execute(sql)
            cur.execute("commit")
        startPoint=endPoint
        parent_id+=2048.0
    print "DONE."
def SnapShot(dt,rdt,nEV,nInterval,nBucket,tab):
    step=(maxID-minID)/nInterval
    for i in range(0,nEV):
        sql="select coordinate[0],coordinate[1] from "+tab+" where id>="+str(startID+i*step)+" and id<"+str(startID+(i+1)*step)

        qs=time.time()
        cur.execute(sql)
        coord=np.array(cur.fetchall())
        qe = time.time()

        if len(coord)<1:
            continue

        ps=time.time()
        OriginalViz=hashByNumpy(coord,r=map,b=ev)
        pe=time.time()

        ss=time.time()
        for x in range(0,ev[0]):
            for y in range(0,ev[1]):
                if OriginalViz[x][y]!=0:
                    sql="insert into "+dt+" values("+str(i)+","+str(x)+","+str(y)+")"
                    cur.execute(sql)
        cur.execute("commit")
        se = time.time()
        #print "Original viz for partition",i,"Done."

        ############################################
        rt=0
        for r in range(0,nBucket):
            ks=int(r*10/100.0*len(coord))
            ke=int((r+1)*10/100.0*len(coord))
            rs=time.time()
            tmpVizA=hashByNumpy(coord[0:ke],r=map,b=ev)
            LowVizA=hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizA))),r=((0,ev[0]),(0,ev[1])),b=rv)
            ###reversed value
            tmpVizB=hashByNumpy(coord[ks-len(coord):],r=map,b=ev)
            LowVizB=hashByNumpy(np.array(np.transpose(np.nonzero(tmpVizB))),r=((0,ev[0]),(0,ev[1])),b=rv)
            re = time.time()
            rt+=re-rs

            for x in range(0,rv[0]):
                for y in range(0,rv[1]):
                    if LowVizA[x][y]!=0 or LowVizB[x][y]!=0:
                        sql="insert into "+rdt+" values("+str(i)+","+str(r)+","+str(x)+","+str(y)+","+str(LowVizA[x][y])+","+str(LowVizB[x][y])+")"
                        cur.execute(sql)
            cur.execute("commit")
            #print "RV for partition",i,"bucket",r, ", Done."
        print qe-qs, pe-ps+rt, se-ss
    print "DONE."


def List2Matrix(list,matrix):
    for i in list:
        matrix[i[0]][i[1]]=i[2]

def RewrittenQuery(lev,rev,yl,yr,xl,xr,etab,rtab):#14963
    sql="select distinct x,y from "+etab+" where viz>="+str(lev)+" and viz<="+str(rev)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_rl=hashByNumpy(np.array(hist),r=((0,ev[0]),(0,ev[1])),b=rv)


    sql="select distinct x,y from "+etab+" where viz>"+str(lev)+" and viz<"+str(rev)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_m = np.zeros(shape=rv)
    if len(hist) > 0:
        RV_m=hashByNumpy(np.array(hist),r=((0,ev[0]),(0,ev[1])),b=rv)

    sql="select x,y,b from "+rtab+" where part="+str(lev)+" and buck="+str(yl)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_yl = np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist,RV_yl)

    sql="select x,y,a from "+rtab+" where part="+str(rev)+" and buck="+str(yr)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_yr=np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist,RV_yr)

    sql="select x,y,b from "+rtab+" where part="+str(lev)+" and buck="+str(xl)
    cur.execute(sql)
    hist = cur.fetchall()
    RV_xl=np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist,RV_xl)

    sql="select x,y,a from "+rtab+" where part="+str(rev)+" and buck="+str(xr)
    cur.execute(sql)
    hist=cur.fetchall()
    RV_xr=np.zeros(shape=rv)
    if len(hist) > 0:
        List2Matrix(hist,RV_xr)

    numerator=0.0
    denominator=0.0

    for i in range(0,rv[0]):
        for j in range(0,rv[1]):
            tmp=max(RV_xl[i][j],RV_xr[i][j],RV_m[i][j])
            numerator+=tmp
            denominator+=min(tmp+RV_yl[i][j]-RV_xl[i][j]+RV_yr[i][j]-RV_xr[i][j],RV_rl[i][j])

    return numerator/denominator


def RewrittenQuery2(lb=8,rb=1):#14963
    sql="select distinct x,y from EV where viz>0 and viz<5"
    cur.execute(sql)
    union=np.array(cur.fetchall())
    unionRV=hashByNumpy(union,r=((0,480),(0,270)),b=rv)

    sql="select x,y,b from RV where part=0 and buck="+str(lb)
    cur.execute(sql)
    left=cur.fetchall()
    for r in left:
        if unionRV[r[0]][r[1]]<r[2]:
            unionRV[r[0]][r[1]]=r[2]

    sql="select x,y,a from RV where part=2 and buck="+str(rb)
    cur.execute(sql)
    right=cur.fetchall()
    for r in right:
        if unionRV[r[0]][r[1]]<r[2]:
            unionRV[r[0]][r[1]]=r[2]
    aprmtVal=np.sum(np.reshape(unionRV,(unionRV.size,)))
    print aprmtVal/11174.0

def QualityCompare(b):
    #startID=862500001267011588
    #endID=startID+1466293030932760
    #sql="select coordinate[0],coordinate[1] from bigtweets where id>=862500001267011588 and id<"+str(endID)
    #cur.execute(sql)
    #coordA=np.array(cur.fetchall())
    #startID=endID
    #endID=startID+1466293030932760
    #sql="select coordinate[0],coordinate[1] from bigtweets where id>="+str(startID)+ " and id<"+str(endID)
    #cur.execute(sql)
    #coordB=np.array(cur.fetchall())
    #coordAB=np.concatenate((coordA,coordB),axis=0)
    #pLen=imageLen(coordAB)
    #for r in range(10,100,10):
    #    k=int(r/100.0*len(coordB))
    #    sLen=imageLen(np.concatenate((coordA,coordB[:k]),axis=0))
    #    print r,sLen,pLen, float(sLen)/pLen
    cnt=0
    sql="select sum(v) from pmatrix where (x,y) not in (select x,y from pmatrix where viz='A_b9_v2') and viz='B_b"+str(b)+"_v2'"
    cur.execute(sql)
    row=cur.fetchall()
    if row is not None and row[0][0] is not None:
        cnt+=row[0][0]
    print 'B-A',cnt
    sql="select sum(v) from pmatrix where (x,y) not in (select x,y from pmatrix where viz='B_b"+str(b)+"_v2') and viz='A_b9_v2'"
    cur.execute(sql)
    row=cur.fetchall()
    if row is not None and row[0][0] is not None:
        cnt+=row[0][0]
    print 'A-B',cnt
    sql="select p1.v,p2.v from pmatrix p1, pmatrix p2 where p1.x=p2.x and p1.y=p2.y and p1.viz='A_b9_v2' and p2.viz='B_b"+str(b)+"_v2'"
    cur.execute(sql)
    rows=cur.fetchall()
    for r in rows:
        if r[0]>r[1]:
            cnt+=r[0]
        else:
            cnt+=r[1]
    print cnt,11857,float(cnt)/11857
    ######

def QualityCompare2():
    startID=862500001267011588
    endID=startID+146629303093276
    sql="select coordinate[0],coordinate[1] from bigtweets where id>=862500001267011588 and id<"+str(endID)
    cur.execute(sql)
    coordA=np.array(cur.fetchall())
    startID=endID
    endID=startID+146629303093276
    sql="select coordinate[0],coordinate[1] from bigtweets where id>="+str(startID)+ " and id<"+str(endID)
    cur.execute(sql)
    coordB=np.array(cur.fetchall())
    coordAB=np.concatenate((coordA,coordB),axis=0)
    pLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB),axis=0))
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.5)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen

    GT=0
    for x in range(0,48):
        for y in range(0,27):
            sql="select sum(v) from pmatrix where (viz='A_b5_v1' or viz='A_b6_v1' or viz='A_b7_v1' or viz='A_b8_v1' or viz='A_b9_v1') and x="+str(x)+" and y="+str(y)
            cur.execute(sql)
            row=cur.fetchall()
            sumV1=0
            if len(row)>0 and row[0][0] is not None:
                sumV1=row[0][0]
                #print sumV1
            sql="select v from pmatrix where viz='B_b9_v2' and x="+str(x)+" and y="+str(y)
            cur.execute(sql)
            row=cur.fetchall()
            Bi=0
            if len(row)>0 and row[0][0] is not None:
                Bi=row[0][0]
                #print Bi
            sql="select count(*) from(select distinct(x,y) from pmatrix where x>="+str(x*10)+" and x<"+str((x+1)*10)+" and y>="+str(y*10)+" and y<"+str((y+1)*10)+" and (viz='B' or viz='A')) t"
            cur.execute(sql)
            AB=cur.fetchall()[0][0]
            GT+=min(sumV1+Bi,AB)
    esLen=0
    for x in range(0,48):
        for y in range(0,27):
            sql="select max(v) from pmatrix where (viz='A_b5_v1' or viz='A_b6_v1' or viz='A_b7_v1' or viz='A_b8_v1' or viz='A_b9_v1' or viz='B_b4_v2') and x="+str(x)+" and y="+str(y)
            cur.execute(sql)
            row=cur.fetchall()
            if len(row)>0 and row[0][0] is not None:
                esLen+=row[0][0]
    print 'Estimated quality', float(esLen)/GT,esLen,GT
def QualityCompare3(b):
    startID=862500001267011588
    endID=startID+1466293030932760
    sql="select coordinate[0],coordinate[1] from bigtweets where id>=862500001267011588 and id<"+str(endID)
    cur.execute(sql)
    coordA=np.array(cur.fetchall())
    startID=endID
    endID=startID+1466293030932760
    sql="select coordinate[0],coordinate[1] from bigtweets where id>="+str(startID)+ " and id<"+str(endID)
    cur.execute(sql)
    coordB=np.array(cur.fetchall())
    coordAB=np.concatenate((coordA,coordB),axis=0)
    pLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB),axis=0))
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.1)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.2)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.3)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.4)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.5)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.7)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.8)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    realLen=imageLen(np.concatenate((coordA[:len(coordA)/2],coordB[:int(len(coordB)*0.9)]),axis=0))
    print 'Real Quality', float(realLen)/pLen,realLen,pLen
    GT=0
    for x in range(0,48):
        for y in range(0,27):
            sql="select sum(v) from pmatrix where (viz='A_b4_v3' or viz='B_b9_v3') and x="+str(x)+" and y="+str(y)
            cur.execute(sql)
            row=cur.fetchall()
            sumV1=0
            if len(row)>0 and row[0][0] is not None:
                sumV1=row[0][0]
            sql="select count(*) from(select distinct(x,y) from pmatrix where x>="+str(x*10)+" and x<"+str((x+1)*10)+" and y>="+str(y*10)+" and y<"+str((y+1)*10)+" and (viz='B' or viz='A')) t"
            cur.execute(sql)
            AB=cur.fetchall()[0][0]
            GT+=min(sumV1,AB)
    esLen=0
    for x in range(0,48):
        for y in range(0,27):
            sql="select max(v) from pmatrix where (viz='A_b6_v3' or viz='B_b"+str(b-1)+"_v2') and x="+str(x)+" and y="+str(y)
            cur.execute(sql)
            row=cur.fetchall()
            if len(row)>0 and row[0][0] is not None:
                esLen+=row[0][0]
    print 'Estimated quality', float(esLen)/GT,esLen,GT

# find k of each day in uber
# quality: map quality, sDate: start date, eDate: end date, step: range of histogram, tab: table name


def KofDay(quality=0.85,sDate='2014-4-1',eDate='2014-9-30',step=1,tab='uber'):
    t1=sDate
    t2=t1
    while(datetime.datetime.strptime(t1,'%Y-%m-%d')+datetime.timedelta(days=step)<=datetime.datetime.strptime(eDate,'%Y-%m-%d')):
        t2=datetime.datetime.strftime(datetime.datetime.strptime(t1,'%Y-%m-%d')+datetime.timedelta(days=step),'%Y-%m-%d')
        sql="select coordinate[1],coordinate[0] from uber where ts>to_date('"+t1+"','yyyy-mm-dd hr24:mi:ss') and ts<=to_date('"+t2+"','yyyy-mm-dd hr24:mi:ss')"
        cur.execute(sql)
        coord=cur.fetchall()
        t1=t2
        pLen=np.count_nonzero(hashByNumpy(np.array(coord),r=((-75, -72), (39, 43))))
        r=1
        sLen=0
        while(sLen<quality*pLen):
            r+=1
            k=int(r/100.0*len(coord))
            sLen=np.count_nonzero(hashByNumpy(np.array(coord[:k]),r=((-75, -72), (39, 43))))
        print t1,r,len(coord),float(sLen)/float(pLen)


def SSIM(X,Y):
    x=X.flatten()
    y=Y.flatten()
    ux=np.mean(x)
    uy=np.mean(y)
    sigmax=np.cov(x)
    sigmay=np.cov(y)
    sigmaxy=np.cov(x,y)[0,1]
    ##
    k1=0.01
    k2=0.03
    L=1
    ##
    C1=(k1*L)**2
    C2=(k2*L)**2
    C3=C2/2
    ##
    l=(2*ux*uy+C1)/(ux**2+uy**2+C1)
    c=(2*math.sqrt(sigmax)*math.sqrt(sigmay)+C2)/(sigmax+sigmay+C2)
    s=(sigmaxy+C3)/(math.sqrt(sigmax)*math.sqrt(sigmay)+C3)
    ##
    return l*c*s
def TestSSIM():
    X=np.arange(12)
    X=X.reshape(3,4)
    Y=np.zeros(12)
    Y=Y.reshape(3,4)
    print SSIM(X,Y)
def VizQuality():
    sql="select distinct x,y from bev"
    cur.execute(sql)
    coord=np.array(cur.fetchall())
    OriginalViz=hashByNumpy(coord,r=((0,480),(0,270)),b=ev)
    sql="select distinct x,y from ev where viz>=0 and viz<4"
    cur.execute(sql)
    coord2=np.array(cur.fetchall())
    AprmtViz=hashByNumpy(coord2,r=((0,480),(0,270)),b=ev)
    print np.count_nonzero(OriginalViz),np.count_nonzero(AprmtViz)
    #print SSIM(OriginalViz,AprmtViz)

#
# max=1412117940 min=1396310400
#


def BiasedQueries(minID=1396310400, maxID=1412117940, nInterval=20):
    size=(maxID-minID)/nInterval
    for s in range(1, nInterval-10):
        for i in range(0, 100):
            ID1 = random.randint(minID+size*10, minID+size*11)
            hisA=IntervalChk(ID=ID1)
            hisB=IntervalChk(ID=ID1+s*size)
            print ID1, ID1+s*size,hisA[0], hisA[1], hisB[0], hisB[1]
# generate queries of length, 1st group: start: 1-3,end:3-5 , 2nd group: start:5-7, end: 7-9
def random_ranges(nInterval):
    f=open("skewed_queries.txt","ab+")
    intervalSize=(maxID-minID)/nInterval
    for i in range(0,1000):
        start=random.randint(minID+intervalSize,minID+3*intervalSize)
        end=start+int(intervalSize*random.uniform(0.5,2.0))
        f.write(str(start)+" "+str(end)+"\n")

    for i in range(0,1000):
        start=random.randint(minID+5*intervalSize,minID+7*intervalSize)
        end=start+int(intervalSize*random.uniform(1.0,2.0))
        f.write(str(start) + " " + str(end) + "\n")

def RandomQueries(nInterval,fn="random_ranges.txt"):
    intervalSize=(maxID-minID)/nInterval
    f=open(fn,'ab+')
    for length in range(1,11):
        for i in range(0, 10):
            start = random.randint(minID , minID +  intervalSize)
            end = int(start + length * intervalSize)
            len = math.floor((end - start) / intervalSize)
            f.write(str(start) + " " + str(end) + " " + str(len) + '\n')



def RandomQuereisForVAS(nInterval, fn):
    f=open(fn,"ab+")
    intervalSize = (maxID - minID) / nInterval
    for i in range(1,7):
        for j in range(0,2):
            start=random.randint(minID+i*intervalSize,minID+i*intervalSize+intervalSize)
            end=start+3*intervalSize
            f.write(str(start)+" "+str(end)+'\n')
    for i in range(1,5):
        for j in range(0,2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end=start+5*intervalSize
            f.write(str(start)+" "+str(end)+'\n')
    for i in range(1,3):
        for j in range(0,2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end=start+7*intervalSize
            f.write(str(start)+" "+str(end)+'\n')
    for i in range(1,2):
        for j in range(0,2):
            start = random.randint(minID + i * intervalSize, minID + i * intervalSize + intervalSize)
            end=start+9*intervalSize
            f.write(str(start)+" "+str(end)+'\n')

def GetIntervals(nInterval, fn="random_ranges.txt", dstFn='random_workload.txt'):
    fo=open(fn)
    f = open(dstFn, "ab+")
    for line in fo.readlines():
        ids=line.split()
        hisA= IntervalChk(nInterval,10,int(ids[0]))
        hisB= IntervalChk(nInterval,10,int(ids[1]))
        f.write(ids[0]+" "+ids[1]+" "+str(hisA[0])+" "+str(hisA[1])+" "+str(hisB[0])+" "+str(hisB[1])+"\n")


def IntervalChk(nInterval, sInterval=10, ID=0):
    IntervalLen = (maxID-minID)/nInterval
    sIntervalLen = IntervalLen/sInterval

    IntervalID = math.floor((ID-minID)/IntervalLen)
    sIntervalID = math.floor((ID-minID-IntervalID*IntervalLen)/sIntervalLen)

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

def quality_alpha_theta(startEV, endEV,tab="EV"):
    sql = "select distinct x,y from "+tab+" where viz>=" + str(startEV) + " and viz<=" + str(endEV)
    cur.execute(sql)
    hist1 = cur.fetchall()
    sql = "select distinct x,y from "+tab+" where viz>" + str(startEV) + " and viz<" + str(endEV)
    cur.execute(sql)
    hist2 = cur.fetchall()
    if len(hist1)>0 and len(hist2)>0:
        quality=len(hist2)/float(len(hist1))
        return quality
    else:
        return 0
def CheckIntervalHead(RV,EV,IntervalFrac=70):
    parentIntervalSize=(maxID-minID)/IntervalFrac
    childIntervalSize=parentIntervalSize/10
    endID=minID + (EV+1)*parentIntervalSize
    startID=endID-RV*childIntervalSize
    return startID, endID

def CheckIntervalTail(RV,EV,IntervalFrac=70):
    parentIntervalSize = (maxID - minID) / IntervalFrac
    childIntervalSize = parentIntervalSize / 10
    startID=minID + EV*parentIntervalSize
    endID=startID+RV*childIntervalSize
    return startID,endID


def quality_alpha_gamma(startEV, endEV, startRV, endRV,etab,rtab):
    quality = RewrittenQuery(startEV, endEV, startRV, endRV, -1, -1,etab,rtab)
    return quality

def quality_alpha_beta_gamma(startEV, endEV, startRV, endRV, tau,etab,rtab):
    alpha_theta = quality_alpha_theta(startEV, endEV,etab)
    alpha_gamma=quality_alpha_gamma(startEV, endEV, startRV, endRV,etab,rtab)
    ##
    quality=0.0
    ##if starts and ends at the same parent interval, then we only need to search from the startRV+1 and ends at the endRV
    if startEV==endEV:
        for xr in range(startRV+1, endRV):
            quality = RewrittenQuery(startEV, endEV, startRV, endRV, -1, xr,etab,rtab)
            if quality >= tau:
                return alpha_theta, alpha_gamma, quality, -1, xr
    ##else we use different search range.
    else:
        for xr in range(0, endRV):
            quality=RewrittenQuery(startEV, endEV, startRV, endRV, -1, xr,etab,rtab)
            if  quality>= tau:
                return alpha_theta,alpha_gamma,quality, -1, xr
        for xl in range(9, startRV, -1):
            quality=RewrittenQuery(startEV, endEV, startRV, endRV, xl, endRV - 1,etab,rtab)
            if  quality >= tau:
                return alpha_theta,alpha_gamma, quality, xl, endRV-1
    return alpha_theta,alpha_gamma, quality,startRV+1,endRV-1

def three_qualities(tau,fn,resFile,etab,rtab):
    file = open(fn)
    resFn=open(resFile,"ab+")
    for line in file.readlines():
        strs = line.split()
        startEV = int(strs[2])
        startRV = int(strs[3])
        endEV = int(strs[4])
        endRV = int(strs[5])
        result = quality_alpha_beta_gamma(startEV, endEV, startRV, endRV, tau,etab,rtab)
        resFn.write(strs[0]+" "+strs[1]+" "+str(result[0])+" "+str(result[1])+" "+str(result[2])+" "+strs[2]+" "+str(result[3])+" "+strs[4]+" "+str(result[4])+"\n")

def alpha2theta(fn,tab,resFile):
    file = open(fn)
    resFn=open(resFile,"ab+")
    for line in file.readlines():
        strs = line.split()
        startEV = int(strs[2])
        endEV = int(strs[4])
        resFn.write(strs[0]+" "+strs[1]+" "+strs[2]+" "+strs[4]+" "+str(quality_alpha_theta(startEV, endEV, tab))+"\n")


def record_count(fn,resFile):
    f=open(fn)
    resFn=open(resFile,"ab+")
    i=0
    for line in f.readlines():
        ids=line.split()
        sql="select count(*) from bigtweets where id between "+ids[0]+" and "+ids[1]
        cur.execute(sql)
        left=cur.fetchall()[0][0]

        sql = "select count(*) from bigtweets where id between " + ids[2] + " and " + ids[3]
        cur.execute(sql)
        right = cur.fetchall()[0][0]
        resFn.write(str(int(left)+int(right))+"\n")
        print int(left)+int(right)
        i+=1
        if i==999:
            return

def QualityDiff():
    IntervalSize = (maxID - minID) / 70
    for length in range(1,6):
        start=int(1.1*IntervalSize+startID)
        end = start + length * IntervalSize
        coord=GetCoordinateRange('taxi', start, end)
        OriginalLen=imageLen(np.array(coord))
        for r in range(0,10):
            subLen=imageLen(np.array(coord[:int(len(coord)/(length*10.0)*((length-1)*10.0+r+1))]))
            print float(subLen)/OriginalLen
        for r in range(0,1):
            print RewrittenQuery(1, 2 + length - 1, 1, 0, -1, r)
        for r in range(9,0,-1):
            print RewrittenQuery(1, 2 + length - 1, 1, 0, r, 0)


def getIntervalTail(id,nInterval):
    intervalSize=(maxID-minID)/nInterval
    return minID+(id+1)*intervalSize


def getIntervalHead(id,nInterval):
    intervalSize=(maxID-minID)/nInterval
    return minID+id*intervalSize


def getChildIntervalHead(parentID,childID,nInterval):
    intervalSize = (maxID - minID) / nInterval
    childIntervalSize=intervalSize/10
    return minID+parentID*intervalSize+childID*childIntervalSize


def getChildIntervalTail(parentID,childID,nInterval):
    intervalSize = (maxID - minID) / nInterval
    childIntervalSize=intervalSize/10
    return minID+parentID*intervalSize+(childID+1)*childIntervalSize


def result_size_ev(nInterval, fn,resFile):
    f=open(fn)
    resFn=open(resFile,"ab+")
    for line in f.readlines():
        evLL = -1
        evLR = -1
        evRL = -1
        evRR = -1
        strs = line.split()
        if float(strs[4])<0.85:
            evLL = strs[0]
            evLR = getIntervalTail(int(strs[2]), nInterval)
            evRL = getIntervalHead(int(strs[3]), nInterval)
            evRR = strs[1]
            # check if interval overlaps or exceeds
        if evRL<=evLR:
            evLR=evRR
            evRL=-1
            evRR=-1
        resFn.write(str(evLL)+" "+str(evLR)+" "+str(evRL)+" "+str(evRR)+"\n")


def result_size_evrv(nInterval, fn="",resFile="",tau=0.85):
    f=open(fn)
    resFn=open(resFile,"ab+")
    for line in f.readlines():
        rvLL = -1
        rvLR = -1
        rvRL = -1
        rvRR = -1
        strs=line.split()
        if float(strs[4])<tau:
            rvLL = strs[0]
            rvLR = getIntervalTail(int(strs[5]), nInterval)
            rvRL = getIntervalHead(int(strs[7]), nInterval)
            rvRR = strs[1]
        elif float(strs[2])<tau and float(strs[4])>=tau:
            if int(strs[6])!=-1:
                rvLL=getChildIntervalHead(int(strs[5]),int(strs[6]),nInterval)
                rvLR=getIntervalTail(int(strs[5]),nInterval)
            if int(strs[8])!=-1:
                rvRL=getIntervalHead(int(strs[7]),nInterval)
                rvRR=getChildIntervalTail(int(strs[7]),int(strs[8]),nInterval)
            # check if interval overlaps or exceeds
            rvLR = min(rvLR, int(strs[1]))
            rvRL = max(rvRL, int(strs[0]))
        if rvRL<=rvLR:
            rvLR=rvRR
            rvRL=-1
            rvRR=-1

        resFn.write(str(rvLL) + " " + str(rvLR) + " " + str(rvRL) + " " + str(rvRR) + "\n")
def stats(evs,rvs,index):
    cntRVEV = 0
    deltaEV = 0
    deltaRV = 0
    for i in range(index*1000,(index+1)*1000-1):
        ev = evs[i].split()
        rv = rvs[i].split()
        delta1 = int(ev[1]) - int(ev[0]) + int(ev[3]) - int(ev[2])
        delta2 = int(rv[1]) - int(rv[0]) + int(rv[3]) - int(rv[2])
        if delta2 > delta1:
            cntRVEV += 1
        # else:
            deltaEV += delta1
            deltaRV += delta2
    return deltaEV,deltaRV,cntRVEV
def compare_id_size(evFile,evrvFile):
    fev=open(evFile)
    frv=open(evrvFile)
    evs=fev.readlines()
    rvs=frv.readlines()
    for i in range(0,6):
        print stats(evs,rvs,i)
def result_size_original():
    f=open("quality_skewed_workload_500.txt")
    for line in f.readlines():
        ids=line.split()
        sql="select count(*) from bigtweets where id between "+ids[0]+" and "+ids[1]
        lmt.cur.execute(sql)
        print lmt.cur.fetchall()[0][0]

def heatmap_histogram(dt,nEV,nInterval,tab):
    step=(maxID-minID)/nInterval
    for i in range(0,nEV):
        sql="select coordinate[0],coordinate[1] from "+tab+" where id>="+str(startID+i*step)+" and id<"+str(startID+(i+1)*step)
        cur.execute(sql)
        coord=np.array(cur.fetchall())
        if len(coord)<1:
            continue
        OriginalViz=hashByNumpy(coord,r=map,b=hv)
        for x in range(0,hv[0]):
            for y in range(0,hv[1]):
                # if OriginalViz[x][y]!=0:
                sql="insert into "+dt+" values("+str(i)+","+str(x)+","+str(y)+","+str(OriginalViz[x][y])+")"
                cur.execute(sql)
        cur.execute("commit")
    print "DONE."


def sample_seek(startID, endID):
    b = StringIO.StringIO()
    c = pycurl.Curl()
    sql='{"select": ["x", "y"],"filters":[{"attribute": "id", "operator": "IN", "operands": ['+str(startID)+', '+str(endID)+']}]}'
    c.setopt(pycurl.URL, 'http://localhost:8080/query')
    c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json','Content-Length:'+str(len(sql))])
    c.setopt(pycurl.CUSTOMREQUEST, "POST")
    c.setopt(pycurl.POSTFIELDS, sql)
    c.setopt(pycurl.WRITEFUNCTION, b.write)
    c.perform()
    c.close()
    val=json.loads(b.getvalue())
    b.close()
    return val["result"]


def sample_vas(nEV, nInterval):
    interval_size=(maxID-minID)/nInterval
    startID=minID
    endID=minID+interval_size*nEV

    sql="select id,coordinate[0],coordinate[1],0,0 from bigtweets where id between "+str(startID)+" and "+str(endID)
    # sql="select id,x,y,0,0 from testVas"
    cur.execute(sql)
    sample=vas.VAS(cur.fetchall())
    for i in range(0,len(sample)):
        sql="insert into vas_sample values("+str(sample[i][0])+",point("+str(sample[i][1])+","+str(sample[i][2])+"))"
        cur.execute(sql)
    cur.execute("commit")

    print "VAS sample created."


def distribution_precision(startID, endID):
    coord1=np.array(sample_seek(startID,endID))
    ss_matrix=hashByNumpy(coord1,map,hv)

    sql="select coordinate[0],coordinate[1] from Bigtweets where ID between "+str(startID)+" and "+str(endID)
    cur.execute(sql)
    original_matrix=hashByNumpy(np.array(cur.fetchall()), map, hv)

    startHV=IntervalChk(500,10,startID)[0]
    endHV = IntervalChk(500, 10, endID)[0]

    sql = "select x,y,sum(density) from HV where viz >" + str(startHV) + " and viz<" + str(endHV)+" group by x,y"
    cur.execute(sql)
    mvs_inner_list=np.array(cur.fetchall())

    sql = "select x,y,sum(density) from HV where viz >=" + str(startHV) + " and viz<=" + str(endHV)+" group by x,y"
    cur.execute(sql)
    mvs_outer_list = np.array(cur.fetchall())


    ss_sum=float(np.sum(ss_matrix))
    original_sum=float(np.sum(original_matrix))
    mvs_inner_sum = float(np.sum(mvs_inner_list[:, 2]))
    mvs_outer_sum = float(np.sum(mvs_outer_list[:, 2]))

    max_diff = 0.0
    max_index_x = 0
    max_index_y = 0
    max_index_i = 0
    for i in range(0,len(mvs_inner_list)):
        if mvs_outer_list[i][2]-mvs_inner_list[i][2] > max_diff:
            max_diff = mvs_outer_list[i][2]-mvs_inner_list[i][2]
            max_index_x=mvs_inner_list[i][0]
            max_index_y = mvs_inner_list[i][1]
            max_index_i = i

    ss_eps=0.0
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ss_eps+=(ss_matrix[i][j]/ss_sum-original_matrix[i][j]/original_sum)**2
    ss_eps=math.sqrt(ss_eps)

    mvs_real_eps=0.0
    for row in mvs_inner_list:
        mvs_real_eps+=(row[2]/mvs_inner_sum-original_matrix[row[0]][row[1]]/mvs_outer_sum)**2
    mvs_real_eps=math.sqrt(mvs_real_eps)

    mvs_est_eps=0.0
    mvs_inner_sum+=max_diff
    mvs_inner_sum-=mvs_inner_list[max_index_i][2]
    mvs_inner_list[max_index_i][2]=max_diff

    for i in range(0,len(mvs_inner_list)):
        mvs_est_eps+=(mvs_inner_list[i][2]/mvs_inner_sum-mvs_outer_list[i][2]/original_sum)**2
    mvs_est_eps=math.sqrt(mvs_est_eps)
    print "sample seek precision:", ss_eps, "mvs real precision:", mvs_real_eps,"mvs est precision:", mvs_est_eps

def test(nEV):
    end=minID+nEV*(maxID-minID)/500
    sql = "select coordinate[0],coordinate[1] from bigtweets where id between " + str(minID)+ " and " + str(end)
    cur.execute(sql)
    len = imageLen(np.array(cur.fetchall()))

    sql="select coordinate[0],coordinate[1] from vas_sample"
    cur.execute(sql)
    vas_len = imageLen(np.array(cur.fetchall()))

    print float(vas_len)/len

def vas_quality(fn):
    f=open(fn)
    for line in f.readlines():
        id=line.split()
        sql="select coordinate[0],coordinate[1] from bigtweets where id between "+id[0]+" and "+id[1]
        cur.execute(sql)
        len=imageLen(np.array(cur.fetchall()))

        sql="select coordinate[0],coordinate[1] from vas_sample where id between "+id[0]+" and "+id[1]
        cur.execute(sql)
        vas_len=imageLen(np.array(cur.fetchall()))

        startInterval=IntervalChk(5000,10,int(id[0]))
        endInterval = IntervalChk(5000, 10, int(id[1]))
        sql="select count(*) from (select distinct x,y from ev where viz>"+str(startInterval[0])+" and viz<"+str(endInterval[0])+") t"
        cur.execute(sql)
        mvs_len=cur.fetchall()[0][0]

        print float(vas_len)/len, float(mvs_len)/len, startInterval[0], endInterval[0]


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
    print e - s, len(coord1),eee-sss,len(coord3), ee - ss, len(coord2)


def dp_comparison(startID, endID):
    coord1=np.array(sample_seek(startID,endID))
    ss_matrix=hashByNumpy(coord1,map,hv)

    sql="select coordinate[0],coordinate[1] from Bigtweets where ID between "+str(startID)+" and "+str(endID)
    cur.execute(sql)
    original_matrix=hashByNumpy(np.array(cur.fetchall()), map, hv)

    startHV=IntervalChk(500,10,startID)[0]
    endHV = IntervalChk(500, 10, endID)[0]

    sql = "select x,y,sum(density) from HV where viz >" + str(startHV) + " and viz<" + str(endHV)+" group by x,y"
    cur.execute(sql)
    mvs_inner_list=np.array(cur.fetchall())
    alpha_matrix=np.zeros(hv)
    for r in mvs_inner_list:
        alpha_matrix[r[0]][r[1]]=r[2]

    sql = "select x,y,sum(density) from HV where viz >=" + str(startHV) + " and viz<=" + str(endHV)+" group by x,y"
    cur.execute(sql)
    mvs_outer_list = np.array(cur.fetchall())
    theta_matrix=np.zeros(hv)
    for r in mvs_outer_list:
        theta_matrix[r[0]][r[1]]=r[2]


    ss_sum=float(np.sum(ss_matrix))
    original_sum=float(np.sum(original_matrix))
    alpha_sum=float(np.sum(alpha_matrix))

    ss_eps=0.0
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ss_eps+=(ss_matrix[i][j]/ss_sum-original_matrix[i][j]/original_sum)**2
    ss_eps = math.sqrt(ss_eps)

    alpha_original_eps=0.0
    for i in range(0, len(alpha_matrix)):
        for j in range(0, len(alpha_matrix[0])):
            alpha_original_eps+=(alpha_matrix[i][j]/alpha_sum-original_matrix[i][j]/original_sum)**2
    alpha_original_eps=math.sqrt(alpha_original_eps)

    alpha_theta_eps=dp_upper_bound(alpha_matrix.flatten(), theta_matrix.flatten(), 10000)

    # print "sample seek precision:", ss_eps, "mvs real precision:", alpha_original_eps,"mvs est precision:", alpha_theta_eps
    print ss_eps,alpha_original_eps, alpha_theta_eps

def dp_comparison_step(startID, endID):
    coord1=np.array(sample_seek(startID,endID))
    ss_matrix=hashByNumpy(coord1,map,hv)

    ssfile=open("ssHeatmap.txt","ab+")
    for i in range(0, len(ss_matrix)):
        for j in range(0,len(ss_matrix[0])):
            ssfile.write(str(i)+" "+str(j)+" "+str(float(ss_matrix[i][j])/np.sum(ss_matrix))+"\n")

    sql="select coordinate[0],coordinate[1] from Bigtweets where ID between "+str(startID)+" and "+str(endID)
    cur.execute(sql)
    original_matrix=hashByNumpy(np.array(cur.fetchall()), map, hv)

    ssfile = open("orHeatmap.txt", "ab+")
    for i in range(0, len(original_matrix)):
        for j in range(0, len(original_matrix[0])):
            ssfile.write(str(i) + " " + str(j) + " " + str(float(original_matrix[i][j])/np.sum(original_matrix)) + "\n")

    startHV=IntervalChk(500,10,startID)[0]
    endHV = IntervalChk(500, 10, endID)[0]



    sql = "select x,y,sum(density) from HV where viz >=" + str(startHV) + " and viz<=" + str(endHV)+" group by x,y"
    cur.execute(sql)
    mvs_outer_list = np.array(cur.fetchall())
    theta_matrix=np.zeros(hv)
    for r in mvs_outer_list:
        theta_matrix[r[0]][r[1]]=r[2]


    ss_sum=float(np.sum(ss_matrix))
    original_sum=float(np.sum(original_matrix))


    ss_eps=0.0
    for i in range(0, len(ss_matrix)):
        for j in range(0, len(ss_matrix[0])):
            ss_eps+=(ss_matrix[i][j]/ss_sum-original_matrix[i][j]/original_sum)**2
    ss_eps = math.sqrt(ss_eps)

    for v in range(2, endHV+1):
        sql = "select x,y,sum(density) from HV where viz >" + str(startHV) + " and viz<" + str(v) + " group by x,y"
        cur.execute(sql)
        mvs_inner_list = np.array(cur.fetchall())
        alpha_matrix = np.zeros(hv)
        for r in mvs_inner_list:
            alpha_matrix[r[0]][r[1]] = r[2]

        alpha_sum = float(np.sum(alpha_matrix))

        alpha_original_eps=0.0
        for i in range(0, len(alpha_matrix)):
            for j in range(0, len(alpha_matrix[0])):
                alpha_original_eps+=(alpha_matrix[i][j]/alpha_sum-original_matrix[i][j]/original_sum)**2
        alpha_original_eps=math.sqrt(alpha_original_eps)

        alpha_theta_eps=dp_upper_bound(alpha_matrix.flatten(), theta_matrix.flatten(), 10000)

        # print "sample seek precision:", ss_eps, "mvs real precision:", alpha_original_eps,"mvs est precision:", alpha_theta_eps
        print ss_eps,alpha_original_eps, alpha_theta_eps