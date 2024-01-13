from django.contrib import admin
from .models import Message,CustomUser,LLMMessages,ConversationHistory,TranscribedAudio, ProblemsTracker
# Register your models here.
admin.site.register(Message)
admin.site.register(CustomUser)
admin.site.register(LLMMessages)
admin.site.register(ProblemsTracker)
admin.site.register(ConversationHistory)
admin.site.register(TranscribedAudio)