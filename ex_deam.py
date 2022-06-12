import deam_prediction
from timeit import default_timer as timer

if __name__ == "__main__":
    start = timer()
    deamPredict = deam_prediction.DeamPredict()
    end = timer()
    print("Time elapsed = ", end - start)
    
    conv = ['Hi!', 'Yo', 'How are You', 'I am fine', 'How is life going']
    print(deamPredict.coherence_score(conv))

    start = timer()
    print(deamPredict.predict("Hi!</UTT>Yo</UTT>How are You</UTT>I am fine</UTT>How is life going</UTT>What are you doing</UTT>What is the meeting about?</UTT>I cannot tell you</UTT>What can you tell me?</UTT>I am god</UTT>I just finished my homework</UTT>"))
    end = timer()
    print("Time elapsed = ", end - start)

    start = timer()
    print(deamPredict.predict("Hi!</UTT>Yo</UTT>How are You</UTT>I am fine</UTT>How is life going</UTT>Busy with meetings</UTT>What is the meeting about?</UTT>I cannot tell you</UTT>What can you tell me?</UTT>Nothing much I am in love</UTT>"))
    end = timer()
    print("Time elapsed = ", end - start)

    start = timer()
    print(deamPredict.predict("hi, how are you doing? i'm getting ready to do some cheetah chasing to stay in shape.</UTT>you must be very fast. hunting is one of my favorite hobbies.</UTT>i am! for my hobby i like to do canning or some whittling.</UTT>i also remodel homes when i am not out bow hunting.</UTT>that's neat. when i was in high school i placed 6th in 100m dash!</UTT>that's awesome. do you have a favorite season or time of year?</UTT>i do not. but i do have a favorite meat since that is all i eat exclusively.</UTT>what is your favorite meat to eat?</UTT>i would have to say its prime rib. do you have any favorite foods?</UTT>i like chicken or macaroni and cheese.</UTT>do you have anything planned for today? i think i am going to do some canning.</UTT>i am going to watch football. what are you canning?</UTT>i think i will can some jam. do you also play footfall for fun.</UTT>"))
    end = timer()
    print("Time elapsed = ", end - start)