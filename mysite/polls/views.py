# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from Code.Evaluate_result.application import *
from Code.main import *
from django.http import HttpResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def check_circ(request):
    if request.method == "POST":
        print("-----------")
        print(request.POST)
        print("-----------")
        rna_list = request.POST.getlist('data')
        model_type = int(request.POST.get('type'))
        result_list = test_best_model(rna_list, model_type)

        return JsonResponse({'results': result_list})


# 上传训练文件
@csrf_exempt
def upload_train_file(request):
    if request.method == "POST":
        f_train = request.FILES.get("train_file", None)
        if f_train:
            print(f_train.name)
            path = cfg.custom_train_data_dir
            f = open(path, 'wb+')
            for chunk in f_train.chunks():
                f.write(chunk)
            f.close()
            return HttpResponse('OK')


# 上传测试文件
@csrf_exempt
def upload_test_file(request):
    if request.method == "POST":
        f_test = request.FILES.get("test_file", None)
        if f_test:
            print(f_test.name)
            path = cfg.custom_test_data_dir
            f = open(path, 'wb+')
            for chunk in f_test.chunks():
                f.write(chunk)
            f.close()
            return HttpResponse('OK')


# 模型训练
@csrf_exempt
def train_model(request):
    if request.method == "POST":
        f_train = cfg.custom_train_data_dir
        model_type = int(request.POST.get('model_type'))
        ratio = float(request.POST.get('ratio'))
        m_criterion = int(request.POST.get('criterion'))
        m_optimizer = int(request.POST.get('optimizer'))
        lr = float(request.POST.get('lr'))
        batch_size = int(request.POST.get('batch_size'))
        shuffle = bool(request.POST.get('shuffle'))
        patience = int(request.POST.get('patience'))
        delta = float(request.POST.get('delta'))
        m_epoch = int(request.POST.get('epoch'))
        params = {'ratio': ratio, 'criterion': m_criterion, 'optimizer': m_optimizer, 'lr': lr, 'batch_size': batch_size,
                  'shuffle': shuffle, 'patience': patience, 'delta': delta, 'epoch': m_epoch}
        train_new_model(f_train, model_type, params)
        return HttpResponse('OK')
    # f_train = cfg.data_dir
    # model_type = 0
    # params = {'ratio': 0.7978, 'criterion': 0, 'optimizer': 0, 'lr': 0.001, 'batch_size': 128, 'shuffle': True,
    #           'patience': 24, 'delta': 0.005, 'epoch': 200}
    # train_new_model(f_train, model_type, params)
    # return HttpResponse('OK')


# 测试模型
@csrf_exempt
def test_model(request):
    if request.method == "POST":
        model_type = int(request.POST.get("model_type"))
        test_result = m_test_model(model_type)
        return JsonResponse(test_result)


# 获取训练进度
@csrf_exempt
def get_progress(request):
    if request.method == "POST":
        current_info = get_current_info()
        return JsonResponse(current_info)
    elif request.method == "GET":
        current_info = get_current_info()
        return render(request, 'polls/info.html', current_info)

# 获取训练结束后的损失函数图像
@csrf_exempt
def loss_image(request):
    if request.method == "POST":
        model_type = int(request.POST.get("model_type"))
        path = get_loss_image_dir(model_type)
        if path != "":
            image_data = open(path, 'rb').read()
            if image_data:
                return HttpResponse(image_data, content_type='image/png')


# 获取测试集的混淆矩阵
@csrf_exempt
def matrix_image(request):
    if request.method == "POST":
        model_type = int(request.POST.get("model_type"))
        path = get_matrix_image_dir(model_type)
        if path != "":
            image_data = open(path, 'rb').read()
            if image_data:
                return HttpResponse(image_data, content_type='image/png')


# 获取测试集的ROC曲线
@csrf_exempt
def roc_image(request):
    if request.method == "POST":
        model_type = int(request.POST.get("model_type"))
        path = get_roc_image_dir(model_type)
        if path != "":
            image_data = open(path, 'rb').read()
            if image_data:
                return HttpResponse(image_data, content_type='image/png')


def test(request):
    return render(request, 'polls/test.html')


def download_train(request):
    file_path = cfg.data_dir
    try:
        response = FileResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response
    except Exception:
        raise Http404("下载文件失败！")


def download_test(request):
    file_path = cfg.test_data_dir
    try:
        response = FileResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response
    except Exception:
        raise Http404("下载文件失败！")


def download_loss(request):
    zip_name = zip_file(0)
    try:
        response = FileResponse(open(zip_name, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(zip_name)
        return response
    except Exception:
        raise Http404("下载文件失败！")


def download_matrix(request):
    zip_name = zip_file(1)
    try:
        response = FileResponse(open(zip_name, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(zip_name)
        return response
    except Exception:
        raise Http404("下载文件失败！")


def download_roc(request):
    zip_name = zip_file(2)
    try:
        response = FileResponse(open(zip_name, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(zip_name)
        return response
    except Exception:
        raise Http404("下载文件失败！")


