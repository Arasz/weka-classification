package put.cs.idss.ml.weka.Comparator;

/**
 * Created by Rafal on 12.01.2017.
 */
public class SimpleTimer {

    long elapsed = 0;
    long started = 0;
    boolean isRunning = false;


    public void start(){
        if(!isRunning){
            isRunning = true;
            started = System.currentTimeMillis();
        }
    }

    public void stop(){
        if(isRunning) {
            isRunning = false;
            elapsed = System.currentTimeMillis() - started;
        }
    }

    public void reset(){
        elapsed = 0;
    }

    public void restart(){
        if(isRunning){
            stop();
            reset();
            start();
        }
    }

    public long getElapsed(){
        return elapsed;
    }

}
