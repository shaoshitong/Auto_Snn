import psutil


def getMemCpu():
    data = psutil.virtual_memory()
    total = data.total  # 总内存,单位为byte
    print('total', total)
    free = data.available  # 可用内存
    print('free', free)

    memory = "Memory usage:%d" % (int(round(data.percent))) + "%" + " "  # 内存使用情况
    print('memory', memory)
    cpu = "CPU:%0.2f" % psutil.cpu_percent(interval=1) + "%"  # CPU占用情况
    print('cpu', cpu)