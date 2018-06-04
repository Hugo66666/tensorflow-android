package com.example.a52641.tensorflow_android;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    static {
        System.loadLibrary("tensorflow_inference");
    }


    private static final String TAG = "Opencv_test" ;

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    private TensorFlowInferenceInterface tensorFlowInferenceInterface = null;
    private static final String mode_file = "file:///android_asset/MnistTF_model.pb";
    private static final String INPUT_NODE = "conv2d_1_input_2:0";
    private static final String OUTPUT_NODE = "dense_3_2/Softmax:0";
    private float[] inputs_data = new float[784];
    private float[] outputs_data = new float[10];




    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(), mode_file);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            //申请权限
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
        }

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(640,640);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat img_rgb = inputFrame.rgba();
        Mat img_t = new Mat();

        Core.transpose(img_rgb,img_t);//转置函数，可以水平的图像变为垂直
        Imgproc.resize(img_t, img_rgb, img_rgb.size(), 0.0D, 0.0D, 0);
        Core.flip(img_rgb, img_rgb,1);  //flipCode>0将mRgbaF水平翻转（沿Y轴翻转）得到mRgba

        Mat img_gray = new Mat();
        Mat img_contours;

        if(img_rgb != null) {
            Imgproc.cvtColor(img_rgb, img_gray, Imgproc.COLOR_RGB2GRAY);

            Imgproc.threshold(img_gray, img_gray, 140, 255, Imgproc.THRESH_BINARY_INV);
            Mat ele1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Mat ele2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));
            Imgproc.erode(img_gray, img_gray, ele1);
            Imgproc.dilate(img_gray, img_gray, ele2);

            img_contours = img_gray.clone();
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(img_contours, contours, new Mat(),
                    Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
                double contourArea = Imgproc.contourArea(contours.get(contourIdx));
                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
                if (contourArea < 1500 || contourArea > 20000)
                    continue;

                Mat roi = new Mat(img_gray, rect);
                Imgproc.resize(roi, roi, new Size(28, 28));

                Bitmap bitmap2 = Bitmap.createBitmap(roi.width(), roi.height(), Bitmap.Config.RGB_565);
                Utils.matToBitmap(roi, bitmap2);
                int number = toNumber(bitmap2);
                if (number >= 0) {
                    //tl左上角顶点  br右下角定点
                    double x = rect.tl().x;
                    double y = rect.br().y;
                    Point p = new Point(x, y);
                    Imgproc.rectangle(img_rgb, rect.tl(), rect.br(), new Scalar(0, 0, 255));
                    Imgproc.putText(img_rgb, Integer.toString(number), p, Core.FONT_HERSHEY_DUPLEX,
                            6, new Scalar(0, 0, 255), 2);
                }
            }
            img_contours.release();
        }
        img_t.release();
        return  img_rgb;
    }


    public int toNumber(Bitmap bitmap_roi){
        int width = bitmap_roi.getWidth();
        int height = bitmap_roi.getHeight();
        int[] pixels = new int[width * height];

        Log.d("tag", width+"  "+height);

        try {
            bitmap_roi.getPixels(pixels, 0, width, 0, 0, width, height);
            for (int i = 0; i < pixels.length; i++) {
                inputs_data[i] = (float)pixels[i];
            }
        }catch (Exception e){
            Log.d("tag", e.getMessage());
        }

        Log.d("Tag", "width: "+width+"   height:"+height);

        Trace.beginSection("feed");
        tensorFlowInferenceInterface.feed(INPUT_NODE, inputs_data, 1,28,28,1);
        Trace.endSection();

        Trace.beginSection("run");
        tensorFlowInferenceInterface.run(new String[]{OUTPUT_NODE});
        Trace.endSection();

        Trace.beginSection("fetch");
        tensorFlowInferenceInterface.fetch(OUTPUT_NODE, outputs_data);
        Trace.endSection();

        int logit = 0;
        for(int i=1;i<10;i++)
        {
            if(outputs_data[i]>outputs_data[logit])
                logit=i;
        }

        if(outputs_data[logit]>0)
            return logit;
        return -1;
    }
}
