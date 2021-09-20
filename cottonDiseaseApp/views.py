from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import cv2
import re
from .predictDisease import imagePred
 
regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'

def home(request):
    if request.method == 'POST' and request.FILES['myfile']:

       
        myfile = request.FILES['myfile']
        print(myfile.size)
        print(myfile.content_type)
        
        if myfile.content_type.split("/")[0] != "image":
            return render(request, 'core/CottonDiseasePrediction.html', {
                'error_file': "Error : Please Upload a Image",
                'uploaded_file_url': ""
            })
        if myfile.size > 23068672:
            return render(request, 'core/CottonDiseasePrediction.html', {
                'error_file': "Error : File size Exceeded 25 MB",
                'uploaded_file_url': ""
            })
       
        try:
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            output_pred = imagePred(uploaded_file_url)
            # attach_file_name = output_file
            # Open the file as binary mode
           
            return render(request, 'core/CottonDiseasePrediction.html', {
                'error_file': output_pred,
                # 'uploaded_file_url': output_file
            })
        except Exception as e:
            return render(request, 'core/CottonDiseasePrediction.html', {
                # 'error_file': "Error : Some Error Occured",
                'error_file': str(e),
                'uploaded_file_url': ""
            })
    return render(request, 'core/CottonDiseasePrediction.html', {
        'uploaded_file_url': ""
    })
