# $LAN=PYTHON$

################################################################
## Copyright (c) 2022 NSYSU CSE M113040037 WU YU-HSUAN 吳宥璇 ##
################################################################


from operator import itemgetter
import random
# from sys import _enablelegacywindowsfsencoding
import tkinter as tk
from tkinter import filedialog
from venv import create
import math
import copy

# the package: tkmacosx is used for Mac os
from tkmacosx import Button

##/////////////////////////////////////////// Data structure and global variable ######################################

# read data
unread = 0
all_data_read = []

points = []

color_list = ["#73B839", "#4798B3", "#FF4D40", "#F08080"] # green, blue, red, brown

step = 0

class VD():

    def __init__(self, origin_point):
        self.p_quantity = len(origin_point)
        self.point = origin_point          # point 數量 = polygon 數量
        self.vertex = []                   # vertex of diagram
        self.edge = []                     # edge of diagram
        self.convexhull = []
        self.upper_line = []
        self.down_line = []
        self.chindex= []
        self.bisector = []


final_VD = VD(points)
all_vd_list = []

##/////////////////////////////////////////// basic function ######################################

def sort_points(points):

    data1 = sorted(points, key=itemgetter(1))
    data2 = sorted(data1, key=itemgetter(0))

    return data2

def sort_points_y(points):
    data1 = sorted(points, key=itemgetter(1))

    return data1
    

def divide(point_list):

    point_list = sort_points(point_list)
    num = len(point_list) // 2
    left_set = point_list[0:num]
    right_set = point_list[num:]

    if check_slope(point_list):
        pass
    else:
        last_l_x = left_set[num-1][0]
        for i in right_set:
            if i[0] == last_l_x:
                left_set.append(i)
                right_set.remove(i)
            elif i[0] > last_l_x:
                break


    return  left_set, right_set


def perpendicular(point_a, point_b):

    xa = point_a[0]
    ya = point_a[1]
    xb = point_b[0]
    yb = point_b[1]

    vertex_a, vertex_b = [], []
    
    # 點重疊
    if xa == xb and ya == yb:
        return point_a, point_a

    elif xa == xb:
        middle_y = (ya + yb) // 2
        return [-100000, middle_y], [100000, middle_y]

    elif ya == yb:
        middle_x = (xa + xb) // 2
        return [middle_x, -100000], [middle_x, 100000]

    else:
        x_center, y_center = (xa+xb)/2, (ya+yb)/2
        slope = -((xa-xb)/(ya-yb))
        # 方程式 = y-y_center = slope(x-x_center)
        # when y=0, 0-y_center = slope(x-x_center)
        # -y_center = slope*x - slope*x_center
        # x = (1/slope)*(-y_center+slope*x_center)
        upper_x = (1/slope)*(-100000-y_center)+x_center
        # when y=600, 600-y_center = slope(x-x_center)
        down_x = (1/slope)*(100000-y_center)+x_center
    
    # 超出邊界 (6種狀況):
    if slope < 0:
        if down_x < -100000 and upper_x > 100000 or upper_x < -100000 and down_x > 100000:
            # y - yc = slope(0-xc)
            # y - yc = slope(600-xc)
            left_y = slope*(-100000-x_center) + y_center
            right_y = slope*(100000-x_center) + y_center
            vertex_a, vertex_b = [-100000, int(left_y)], [100000, int(right_y)] 
        elif down_x < -100000:
            left_y = slope*(-100000-x_center) + y_center
            vertex_a, vertex_b = [int(upper_x), -100000], [-100000, int(left_y)]
        elif upper_x > 100000:
            right_y = slope*(100000-x_center) + y_center
            vertex_a, vertex_b = [100000, int(right_y)], [int(down_x), 100000]
        else:
            vertex_a, vertex_b = [int(upper_x), -100000], [int(down_x), 100000]
    
    else:
        if upper_x < -100000 and down_x > 100000:
            left_y = slope*(-100000-x_center) + y_center
            right_y = slope*(100000-x_center) + y_center
            vertex_a, vertex_b = [-100000, int(left_y)], [100000, int(right_y)]
        elif upper_x < -100000:
            left_y = slope*(-100000-x_center) + y_center
            vertex_a, vertex_b = [-100000, int(left_y)], [int(down_x), 100000]
        elif down_x > 100000:
            right_y = slope*(100000-x_center) + y_center
            vertex_a, vertex_b = [int(upper_x), -100000], [100000, int(right_y)]
        else:
            vertex_a, vertex_b = [int(upper_x), -100000], [int(down_x), 100000]
    
    # vertex_a 要比 vertex_b 還上面
    if vertex_a[1] > vertex_b[1]:   # if vertex_a 在下面 和 vertex_b 交換
            temp = vertex_a
            vertex_a = vertex_b
            vertex_b = temp

    return vertex_a, vertex_b

    
def construction(list):

    if len(list) == 2:

        vertex_a, vertex_b = perpendicular(list[0], list[1])

        new_vd = VD(list)
        new_vd.vertex.append(vertex_a)
        new_vd.vertex.append(vertex_b)
        new_vd.edge.append([vertex_a[0], vertex_a[1], vertex_b[0], vertex_b[1]])
        new_vd.convexhull = [i for i in list]
    
    else:
        new_vd = VD(list)
        new_vd.convexhull = [i for i in list]
        
    return new_vd


def draw_dot(point, c="black", size=2):
    x, y = point[0], point[1]
    x1, y1 = x-size, y+size
    x2, y2 = x+size, y-size
    canvs.create_oval(x1, y1, x2, y2, fill=c, outline=c)


def draw_line(point_a, point_b, c="black", d=()):
    xa = point_a[0]
    ya = point_a[1]
    xb = point_b[0]
    yb = point_b[1]
    canvs.create_line(xa ,ya ,xb ,yb , fill=c, dash=d)


def draw_line_byline(line, c="black", d=()):
    canvs.create_line(line[0] ,line[1] ,line[2] ,line[3] , fill=c, dash=d)


def draw_polygon(outside_points_list, d=(), c="black"):
    if len(outside_points_list) == 1:
        pass
    elif len(outside_points_list) == 2:
        draw_line(outside_points_list[0], outside_points_list[1], c=c, d=d)
    else:
        x_y = []
        for i in outside_points_list:
            x_y.append(i[0])
            x_y.append(i[1])

        canvs.create_polygon(x_y, outline=c, fill="", dash=d, width=2)


def draw_bisector(points_list, d=(), c="red"):
    x_y = []
    for i in points_list:
        x_y.append(i[0])
        x_y.append(i[1])

    canvs.create_line(x_y, fill=c, dash=d)


def draw_vd(VD, c="black"):
    for p in VD.point:
        draw_dot(p, c)
    if len(VD.edge) > 0:
        for e in VD.edge:
            draw_line_byline(e, c)


def clean_canvas():
    global step, num, mergeee, i
    canvs.delete("all")
    points.clear()
    all_vd_list.clear()
    step = 0
    i = 0


##/////////////////////////////////////////// calculate ######################################

''' vector: p1->p2 '''
def vector(p1, p2):
    return [(p2[0] - p1[0]), (p2[1] - p1[1])]


''' points [x, y] [a, b] => edge [x, y, a, b]'''
def return_edge(p1, p2):
    if p1[0] == p2[0]:
        if p1[1] < p2[1]:
            return [p1[0], p1[1], p2[0], p2[1]]
        else:
            return [p2[0], p2[1], p1[0], p1[1]]
    else:
        if p1[0] < p2[0]:
            return [p1[0], p1[1], p2[0], p2[1]]
        else:
            return [p2[0], p2[1], p1[0], p1[1]]

def edge_to_point(edge):
    return [edge[0], edge[1]], [edge[2], edge[3]]

''' 外積 '''
# p1->p2 x p1->p3
# 外積小於0 順時針轉
def cross_calculate(p1, p2, p3):
    return ((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))


'''求兩線段是否相交 line_1 (p1, p2)  line_2 (p3, p4)'''
def line_cross(p1, p2, p3, p4):
    c1 = cross_calculate(p1, p2, p3)
    c2 = cross_calculate(p1, p2, p4)
    c3 = cross_calculate(p3, p4, p1)
    c4 = cross_calculate(p3, p4, p2)

    a = vector(p1, p2)
    b = vector(p3, p4)
    s = vector(p1, p3)

    # point = p1 + a * cross(s, b) / cross(a, b)

    # 三點不共線
    if c1*c2 < 0 and c3*c4 < 0:
        return True
    else:
        return False

''' 中點 '''
def middle(p1, p2):
    return [(p1[0]+p2[0])//2, (p1[1]+p2[1])//2]


''' 點是不是在線上 '''
def point_on_line(p, line_p1, line_p2):
    v1 = vector(p, line_p1)
    v2 = vector(p, line_p2)
    if dot(v1, v2) <=0:
        return True
    else:
        return False


''' 內積 '''
def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]


''' dot(v1, v2) : =0 直角, >0 銳角, <0 鈍角 '''
def angel(p1, p2, p3):
    
    # p1 angle
    v1 = vector(p1, p2)
    v2 = vector(p1, p3)
    
    # p2 angle
    v3 = vector(p2, p1)
    v4 = vector(p2, p3)

    # p3 angle
    v5 = vector(p3, p1)
    v6 = vector(p3, p2)

    return dot(v1, v2), dot(v3, v4), dot(v5, v6)


''' 直角三角形 '''
def is_acute(p1, p2, p3):
    a, b ,c = angel(p1, p2, p3)
    if a*b*c == 0:
        return True
    else:
        return False


''' 鈍角三角形 '''
def is_obtuse(p1, p2, p3):
    a, b ,c = angel(p1, p2, p3)
    if a < 0 or b < 0 or c < 0:
        return True
    else:
        return False 


def edge_length(p1, p2):
    return int(math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2))


''' 鈍角三角形擦線修正 '''
def change_vertex(v1, va, vb):
    if v1 == va:
        return vb
    else:
        return va


''' 斜率 '''
def slope(point_a, point_b):
    if point_a[0] == point_b[0] or point_a[1] == point_b[1]:
        return 0       # 垂直線或水平線
    else:
        return (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])


''' check 平行 or 共線 '''
def check_slope(point_list):
    
    l = len(point_list)
    s = slope(point_list[0], point_list[-1])
    same = True

    for i in range(l-1):
        if slope(point_list[i], point_list[i+1]) == s:
            pass
        else:
            same = False
    
    return same


'''平行 or 共線 voronoi data'''
def parallel(point_list):

    l = len(point_list)
    vertex_list = []
    edge_list = []

    for i in range(l-1):
        a, b = perpendicular(point_list[i], point_list[i+1])
        vertex_list.append(a)
        vertex_list.append(b)
        edge_list.append(return_edge(a, b))
    
    return vertex_list, edge_list


''' 求兩線交點 p1, p2 is a line, p3 p4 is a line
    equation = ax + by = c '''
def cross_point(p1, p2, p3, p4):
    if p2[0]-p1[0] == 0:
        x = p1[0]
        y = (p1[0]-p3[0])/(p4[0]-p3[0])*(p4[1]-p3[1])+p3[1]
    else:
        k1 = (p2[1]-p1[1])*1.0/(p2[0]-p1[0])
        b1 = p1[1]*1.0 - p1[0]*k1*1.0
        if (p4[0]-p3[0]) == 0:
            k2 = None
            b2 = 0
        else:
            k2 = (p4[1]-p3[1])*1.0 / (p4[0]-p3[0])
            b2 = p3[1]*1.0 - p3[0]*k2*1.0
        if k2 == None:
            x = p3[0]
        else:
            x = (b2-b1)*1.0 / (k1-k2)
        y=k1*x*1.0+b1*1.0

    return [int(x), int(y)]


''' 找右側的點 '''
# A-> B, find_ch_vetex(A, B, point)
def is_right(pa, pb, p):
    find = False
    x0, y0 = p[0], p[1]
    x1, y1 = pa[0], pa[1]
    x2, y2 = pb[0], pb[1]
    # p 在 pa->pb 的右側
    if ((x2-x1)*(y0-y1) - (y2-y1)*(x0-x1)) < 0:
        find = True
    
    return find

''' 找左側的點 '''
def is_left(pa, pb, p):
    find = False
    x0, y0 = p[0], p[1]
    x1, y1 = pa[0], pa[1]
    x2, y2 = pb[0], pb[1]
    # p 在 pa->pb 的左側
    if ((x2-x1)*(y0-y1) - (y2-y1)*(x0-x1)) > 0:
        find = True
    
    return find

''' 找convex_hull '''
def find_CH(vd):
    # all_ray = []
    # all_ray_sorted = []
    ch = []
    if vd.p_quantity == 1 or vd.p_quantity == 2:
        ch = vd.point
    elif vd.p_quantity == 3:
        ch.append(vd.point[0])
        # 順時針存
        if cross_calculate(vd.point[0], vd.point[1], vd.point[2]) > 0:
            ch.append(vd.point[1])
            ch.append(vd.point[2])
        else:
            ch.append(vd.point[2])
            ch.append(vd.point[1])
    
    return ch
            

''' 合併 convex hull '''
# 假設CH的點按順時針儲存
def merge_CH(lch, rch):

    global final_VD

    # 找左CH最右點
    ch_index = []
    rightest = [0, 0]
    r_index = 0
    lch_len = len(lch)
    # 找右CH最左點
    leftest = [600, 600]
    l_index = 0
    rch_len = len(rch)
    for i, point in enumerate(lch):
        if point[0] > rightest[0]:
            rightest = point
            r_index = i
    for i, point in enumerate(rch):
        if point[0] < leftest[0]:
            leftest = point
            l_index = i

    upper_a = r_index
    upper_b = l_index

    while(True):
        prev_a = upper_a
        prev_b = upper_b

        # 移動右CH的點找切線
        while is_right(lch[upper_a], rch[upper_b], rch[(upper_b+1)%rch_len]):
            upper_b += 1
            upper_b = upper_b % rch_len
        # 移動左CH的點找切線
        while is_left(rch[upper_b], lch[upper_a], lch[(upper_a+lch_len-1)%lch_len]):
            upper_a = upper_a + lch_len - 1
            upper_a = upper_a % lch_len
        
        if (upper_a == prev_a and upper_b == prev_b):
            break

    lower_a = r_index
    lower_b = l_index

    while(True):
        prev_c = lower_a 
        prev_d = lower_b

        while is_left(lch[lower_a], rch[lower_b], rch[(lower_b+rch_len-1)%rch_len]):
            lower_b = lower_b+rch_len-1
            lower_b = lower_b % rch_len
        while is_right(rch[lower_b], lch[lower_a], lch[(lower_a+1)%lch_len]):
            lower_a += 1
            lower_a = lower_a % lch_len
        
        if (prev_c == lower_a and prev_d == lower_b):
            break 
    

    upper_tangent = [lch[upper_a], rch[upper_b]]
    lower_tangent = [lch[lower_a], rch[lower_b]]
    ch_index = [upper_a, upper_b, lower_a, lower_b]
    final_ch = []

    index = upper_b
    while(True):
        final_ch.append(rch[index])
        if (index == lower_b):
            break
        index = (index + 1) % rch_len

    index = lower_a
    while(True):
        final_ch.append(lch[index])
        if (index == upper_a):
            break
        index = (index + 1) % lch_len

    
    return upper_tangent, lower_tangent, final_ch, ch_index


''' 換ray '''
def find_next_point(point_list, line, origin_point):
    
    l = [[line[0], line[1]], [line[2], line[3]]]
    for p in point_list:
        check = round(slope(l[0], l[1]) * slope(p, origin_point), 3)
        if p != origin_point:
            # print(f"check:{p}")
            # print(f"slope minus={check}")
            if check < -0.9 and check > -1.1:
                return p
            elif slope(l[0], l[1]) == 0 and slope(p, origin_point) == 0:
                return p

''' 找 bisector '''
def find_bisector(l_VD, r_VD, final_VD):
    # print(f"lvd ch = {l_VD.convexhull}")
    # print(f"rvd ch = {r_VD.convexhull}")
    bisector = []
    # draw_line(final_VD.upper_line[0], final_VD.upper_line[1], c="purple")
    # base line: 左CH和右CH相連的線
    base_a, base_b = final_VD.upper_line[0], final_VD.upper_line[1]
    last_point_a, last_point_b = perpendicular(final_VD.down_line[0], final_VD.down_line[1])
    ray_a, ray_b = perpendicular(final_VD.upper_line[0], final_VD.upper_line[1])

    bisector.append(ray_a)

    final_VD.edge = l_VD.edge + r_VD.edge
    l_edge = copy.deepcopy(l_VD.edge)
    r_edge = copy.deepcopy(r_VD.edge)

    final_VD.edge = l_edge + r_edge

    # print(f"start basea={base_a}, baseb={base_b}")
    # print(f"end basea={final_VD.down_line[0]}, baseb={final_VD.down_line[1]}")

    while(True):

        # 先碰到線的點
        l_first = [10000, 10000]
        r_first = [10000, 10000]

        if len(l_edge) > 0:
            for i, edge in enumerate(l_edge):
                if line_cross(ray_a, ray_b , [edge[0], edge[1]], [edge[2], edge[3]]):
                    cp = cross_point(ray_a, ray_b , [edge[0], edge[1]], [edge[2], edge[3]])
                    if cp[1] < l_first[1]:
                        # 新找到的點要比上個點低
                        # if cp[1] > bisector[-1][1]+1:
                        l_first = cp
                        l_ci = i
                    
        if len(r_edge) > 0:
            for i, edge in enumerate(r_edge):
                if line_cross(ray_a, ray_b , [edge[0], edge[1]], [edge[2], edge[3]]):
                    cp = cross_point(ray_a, ray_b , [edge[0], edge[1]], [edge[2], edge[3]])
                    if cp[1] < r_first[1]:
                        # 新找到的點要比上個點低
                        # if cp[1] > bisector[-1][1]+1:
                        r_first = cp
                        r_ci = i
   
        if l_first != [10000, 10000] or r_first != [10000, 10000]:
        # r 在上方
            if r_first[1] < l_first[1]:
                # bisector
                bisector.append(r_first)
                neww = find_next_point(r_VD.point, r_edge[r_ci], base_b)          
                if neww == None:
                    break
                else:
                   base_b = neww 
                # print(f"change base_b = {base_b}")
                # 擦右邊的線
                if is_left(ray_a, ray_b, [r_edge[r_ci][0], r_edge[r_ci][1]]):
                    final_VD.edge.remove(r_edge[r_ci])
                    new_edge = return_edge([r_edge[r_ci][2], r_edge[r_ci][3]], r_first)
                    final_VD.edge.append(new_edge)
                else:
                    final_VD.edge.remove(r_edge[r_ci])
                    new_edge = return_edge([r_edge[r_ci][0], r_edge[r_ci][1]], r_first)
                    final_VD.edge.append(new_edge)
                # 繼續找 bisector
                ray_a, ray_b = perpendicular(base_a, base_b)
                ray_a = r_first
                r_edge.remove(r_edge[r_ci])
                r_edge.append(new_edge)

            else:
                bisector.append(l_first)
                neww = find_next_point(l_VD.point, l_edge[l_ci], base_a)
                if neww == None:
                    break
                else:
                    base_a = neww
                # print(f"change base_a = {base_a}")
                # 擦左邊的線
                if is_right(ray_a, ray_b, [l_edge[l_ci][0], l_edge[l_ci][1]]):
                    final_VD.edge.remove(l_edge[l_ci])
                    new_edge = return_edge([l_edge[l_ci][2], l_edge[l_ci][3]], l_first)
                    final_VD.edge.append(new_edge)
                else:
                    final_VD.edge.remove(l_edge[l_ci])
                    new_edge = return_edge([l_edge[l_ci][0], l_edge[l_ci][1]], l_first)
                    final_VD.edge.append(new_edge)
                # 繼續找 bisector
                ray_a, ray_b = perpendicular(base_a, base_b)
                ray_a = l_first
                l_edge.remove(l_edge[l_ci])
                l_edge.append(new_edge)

            
        # if (base_a == final_VD.down_line[0]) and (base_b == final_VD.down_line[1]):
        # 找不到點
        if l_first == [10000, 10000] and r_first == [10000, 10000]:
            bisector.append(last_point_b)
            break
    
    for i in range(len(bisector)-1):
        e = return_edge(bisector[i], bisector[i+1])
        final_VD.edge.append(e)
    
    # print(bisector)

    # print(f"all fvd edge={final_VD.edge}") 
    # print(f"all l_edge={l_edge}")
    # print(f"all R_edge={r_edge}")
    

    # 檢查所有左VD的的edge都在hyperplane的左邊，右VD的edge都在hyperplane的右邊
    for edge in l_edge:
        find_l = False
        a, b = edge_to_point(edge)
        # 只要 check 最左邊的點 
        if a[0] < b[0]:
            check = a
        if a[0] > 0 :
            for i in range(len(bisector)-1):
                test_line_p1, test_line_p2 = [0, a[1]], a
                if line_cross(test_line_p1, test_line_p2, bisector[i], bisector[i+1]):
                    find_l = True
        if find_l:
            final_VD.edge.remove(edge)
    
    for edge in r_edge:
        find_r = False
        c, d = edge_to_point(edge)
        if c[0] < d[0]:
            check = d
        if d[0] < 600 :
            for i in range(len(bisector)-1):
                test_line_p1, test_line_p2 = d, [600, d[1]]
                if line_cross(test_line_p1, test_line_p2, bisector[i], bisector[i+1]):
                    find_r = True
        if find_r:

            final_VD.edge.remove(edge)


    return bisector

          
    
##/////////////////////////////////////////// main function ######################################

def merge(lvd, rvd):
    
    new_vd = VD(lvd.point + rvd.point)
    new_vd.upper_line, new_vd.down_line, new_vd.convexhull, new_vd.chindex = merge_CH(lvd.convexhull, rvd.convexhull)
    # draw_polygon(new_vd.convexhull, d=(1, 1), c = color_list[2])
    new_vd.bisector = find_bisector(lvd, rvd, new_vd)

    return new_vd



def create_VD(list):


    if len(list) <= 2:
        voronoi_diagram = construction(list)
    else:
        l, r = divide(list)
        lvd = create_VD(l)
        rvd = create_VD(r)
        voronoi_diagram = merge(lvd, rvd)
        set = []
        set.append(lvd)
        set.append(rvd)
        set.append(voronoi_diagram)
        all_vd_list.append(set)


    return voronoi_diagram


def run():

    global points, final_VD

    if len(points) == 0 or len(points) == 1:
        pass
    else:
        if len(points) == 2:
            final_VD = construction(points)
        else:
            final_VD = create_VD(points)
    
    print("--------------------")


def finish():

    global final_VD

    run()
    canvs.delete("all")
    draw_vd(final_VD)    


step = 0
i = 0

def step_by_step():

    global points, all_vd_list, step, i

    if i == 0 and step == 0:
        run()

    merge_times = len(all_vd_list)

    if i < merge_times:

        Lvd = all_vd_list[i][0]
        Rvd = all_vd_list[i][1]
        Mvd = all_vd_list[i][2]

        if step == 0:
            draw_vd(Lvd, c=color_list[0])
            step+=1
        elif step == 1:
            draw_vd(Rvd, c=color_list[1])
            step+=1
        elif step == 2:
            draw_polygon(Lvd.convexhull, c=color_list[0], d=(1, 1))
            draw_polygon(Rvd.convexhull, c=color_list[1], d=(1, 1))
            draw_polygon(Mvd.convexhull, c=color_list[3], d=(1, 1))
            step+=1
        elif step == 3:
            draw_bisector(Mvd.bisector, c=color_list[2])
            step+=1
        elif step == 4:
            canvs.delete("all")
            draw_vd(Mvd)
            step = 0
            i+=1
            




##/////////////////////////////////////////// IO ######################################

def open_file():

    global unread, all_data_read, points

    clean_canvas()
    file_path = tk.filedialog.askopenfilename()
    fo = open(file_path, "r")
    lines = fo.readlines()
    f = []
    for line in lines:
        if line == '\n':
            pass
        else:
            l = line.strip("\n")
            f.append(l)

    one_set = []
    read_data = 0
    all_data_read = []
    
    # f = ['#雙點測試', '2', '289 290', '342 541', '#雙點測試 水平', '2', '200 200', '400 200']
    for content in f:

        if content[0] == "#":
            pass

        elif read_data == 0:
            if int(content) == 0:
                break
            else:
                if len(one_set) > 0:
                    all_data_read.append(one_set)
                    one_set = []
                read_data = int(content)

        else:
            data = content.split(" ")
            data_x = int(data[0])
            data_y = int(data[1])
            one_set.append([data_x, data_y])
            read_data -= 1
        
             
    all_data_read.append(one_set)

    for data in all_data_read[0]:
        draw_dot(data)
        print(f"{data[0]} {data[1]}")
    points = all_data_read[0]    
    unread = len(all_data_read)-1


def next_set():
    
    global unread, all_data_read, points

    clean_canvas()
    set_num = len(all_data_read)

    if unread > 0:
        points = all_data_read[(set_num-unread)]
        for data in all_data_read[(set_num-unread)]:
            draw_dot(data)
            print(f"{data[0]} {data[1]}")
        unread -= 1
    else:
        canvs.create_text(300, 300, text="no more data")


# output 前先做排序
def output_file():

    global final_VD

    file = tk.filedialog.asksaveasfile(filetypes=(("txt files", "*.txt"),("all files", "*.*")))
    f = open(file.name, "w")

    data1 = sorted(final_VD.point, key=itemgetter(1))
    point = sorted(data1, key=itemgetter(0))

    data2 = sorted(final_VD.edge, key=itemgetter(3))
    data3 = sorted(data2, key=itemgetter(2))
    data4 = sorted(data3, key=itemgetter(1))
    edge = sorted(data4, key=itemgetter(0))

    for p in point:
        f.write(f"P {str(p[0])} {str(p[1])}\n")

    for e in edge:
        f.write(f"E {str(e[0])} {str(e[1])} {str(e[2])} {str(e[3])}\n")
    
    f.close()


# read output file
def read():
    clean_canvas()
    file_path = tk.filedialog.askopenfilename()
    fo = open(file_path, "r")
    lines = fo.readlines()
    f = []
    for line in lines:
        l = line.strip("\n")
        f.append(l)
    
    for content in f:
        if content[0] == "P":
            p = content.split(" ")
            draw_dot([int(p[1]), int(p[2])])
        else:
            e = content.split(" ")
            draw_line([int(e[1]), int(e[2])], [int(e[3]), int(e[4])])


##/////////////////////////////////////////// GUI ######################################

back_color = "#FFDAB9"
canv_color = "#FFFFFF"
butt_color = "#FF6347"  
word ="#FFFFFF"

screen = tk.Tk()
screen.title("Voronoi diagram")
screen.geometry('815x700')
screen.config(background=back_color)
canvs = tk.Canvas(screen, width=600, height=600, background=canv_color)
canvs.grid(row=1, column=0, columnspan=7, padx=20, pady=10)

def click_position(event):
    x, y = event.x, event.y
    x1, y1 = x-2, y+2
    x2, y2 = x+2, y-2
    points.append([x, y])
    print(f"{x} {y}")
    canvs.create_oval(x1, y1, x2, y2, fill="black", outline="black")

canvs.bind("<Button-1>", click_position)

# on Mac os (Only different on button setting) !!!

button_run = Button(screen, text="Run", width=100, bg=butt_color, fg=word, command=finish, borderless=1)
button_run.grid(row=0, column=0, padx=15, pady=15)

button_step = Button(screen, text="Step by Step", width=100, background=butt_color, fg=word, command=step_by_step, borderless=1)
button_step.grid(row=0, column=1, padx=15, pady=15)

button_file = Button(screen, text="Read Data", width=100, background=butt_color, fg=word, command=open_file, borderless=1)
button_file.grid(row=0, column=2, padx=15, pady=15)

button_next = Button(screen, text="Next", width=100, background=butt_color, fg=word, command=next_set, borderless=1)
button_next.grid(row=0, column=3, padx=15, pady=15)

button_read = Button(screen, text="Diagram", width=100, background=butt_color, fg=word, command=read, borderless=1)
button_read.grid(row=0, column=4, padx=15, pady=15)

button_save = Button(screen, text="Save", width=100, background=butt_color, fg=word, command=output_file, borderless=1)
button_save.grid(row=0, column=5, padx=15, pady=15)

button_clean = Button(screen, text="Clean", width=100, background=butt_color, fg=word, command=clean_canvas, borderless=1)
button_clean.grid(row=0, column=6, padx=15, pady=15)


# on windows os !!!
# button_run = tk.Button(screen, text="Run", width=11, bg=butt_color, fg=word, command=finish)
# button_run.grid(row=0, column=0, padx=15, pady=15)

# button_step = tk.Button(screen, text="Step by Step", width=11, background=butt_color, fg=word, command=step_by_step)
# button_step.grid(row=0, column=1, padx=15, pady=15)

# button_file = tk.Button(screen, text="Read Data", width=11, background=butt_color, fg=word, command=open_file)
# button_file.grid(row=0, column=2, padx=15, pady=15)

# button_next = tk.Button(screen, text="Next", width=11, background=butt_color, fg=word, command=next_set)
# button_next.grid(row=0, column=3, padx=15, pady=15)

# button_read = tk.Button(screen, text="Diagram", width=11, background=butt_color, fg=word, command=read)
# button_read.grid(row=0, column=4, padx=15, pady=15)

# button_save = tk.Button(screen, text="Save", width=11, background=butt_color, fg=word, command=output_file)
# button_save.grid(row=0, column=5, padx=15, pady=15)

# button_clean = tk.Button(screen, text="Clean", width=10, background=butt_color, fg=word, command=clean_canvas)
# button_clean.grid(row=0, column=6, padx=15, pady=15)

screen.mainloop()





