from django.db import models

# Create your models here.
import datetime
from django.db import models
from django.utils import timezone
from django.contrib import admin


class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    @admin.display(
        boolean=True,
        ordering='pub_date',
        description='Published recently?',
    )
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text


class User(models.Model):
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=32)
    permission = models.SmallIntegerField()


class Result(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    epoch = models.SmallIntegerField()
    pub_date = models.DateTimeField('最后预测时间', auto_now=True)
