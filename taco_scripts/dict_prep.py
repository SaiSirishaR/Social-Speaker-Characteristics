text_array = []
d = {}
#d['SOS'] = len(d)+0
i=1
text_file = open('/home/srallaba/projects/personality_stuff/voices/cmu_us_LJspeech/Speech_Expts_Barebones/taco_25October_expts/Data/cmu_us_bdl_arctic/etc/new_txt.done.data')
for line in text_file:
#  text_array.append(line.split('\n')[0])
#print("text array is", text_array)
  wordseq = line.split('\n')[0]
  words = wordseq.split(' ')
  #print("words are:", words)
  for l in range(0,len(words)-1): 
   if words[l] not in d.keys():
    d[words[l]] = i
    i = i+1

#  d['EOS'] = len(d)+1
print(d['slightly'])
