from django.urls import path
from . import views

app_name = 'polls'
urlpatterns = [
    path('check/', views.check_circ, name='check'),
    path('train_file/', views.upload_train_file, name='train_file'),
    path('test_file/', views.upload_test_file, name='test_file'),
    path('train_model/', views.train_model, name='train_model'),
    path('result/', views.get_progress, name='result'),
    path('loss/', views.loss_image, name='loss'),
    path('test_model/', views.test_model, name='test_model'),
    path('matrix/', views.matrix_image, name='matrix'),
    path('roc/', views.roc_image, name='roc'),
    path('down_train/', views.download_train, name='down_train'),
    path('down_test/', views.download_test, name='down_test'),
    path('down_loss/', views.download_loss, name='down_loss'),
    path('down_matrix/', views.download_matrix, name='down_matrix'),
    path('down_roc/', views.download_roc, name='down_roc'),
]
