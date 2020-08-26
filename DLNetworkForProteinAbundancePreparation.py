import sys, getopt, csv

def is_blank(s):
    return not bool(s and s.strip())

def usage():
   print ('DLNetworkForProteinAbundancePreparation.py --i1 <inputfile1> --i2 <inputfile2> --o1 <outputfile1> --o2 <outputfile2> --d1 <delimiter1> --d2 <delimiter2> --id <identifier> --dout <out_delimiter> --cs --subst --nthr <threshold>')

def main(argv):
   casesensitive = False
   inputfile1 = ''
   inputfile2 = ''
   outputfile1 = ''
   outputfile2 = ''
   delimiter1 = '\t'
   delimiter2 = '\t'
   default_out_delimiter_value = "DeFaUltVaLuE"
   out_delimiter = default_out_delimiter_value
   identifier = 'Id'
   casesensitive = False
   normalize = False
   substitute = False
   norm_threshold = 0.0
   try:
      opts, args = getopt.getopt(argv,"hi:o:d:I:O:D:e:f:c:n:S",["help=","i1=","o1=","d1=","i2=","o2=","d2=","id=","dout=","cs","nthr=","subst"])
   except getopt.GetoptError as err:
      print (err.msg)
      usage()
      sys.exit(2)
   print(opts)
   
   for opt, arg in opts:
      if opt in ("-h","--help"):
         usage()
         sys.exit()
      elif opt in ("-i", "--i1"):
         inputfile1 = arg
         print ('=> Input file 1 is "',inputfile1,'"',sep='')
      elif opt in ("-I", "--i2"):
         inputfile2 = arg
         print ('=> Input file 2 is "',inputfile2,'"',sep='')
      elif opt in ("-o", "--o1"):
         outputfile1 = arg
         print ('=> Output file 1 is "',outputfile1,'"',sep='')
      elif opt in ("-O", "--o2"):
         outputfile2 = arg
         print ('=> Output file 2 is "',outputfile2,'"',sep='')
      elif opt in ("-d", "--d1"):
         if arg != "TAB":
           delimiter1 = arg
         print ('=> Delimiter for file 1 is "',delimiter1,'"',sep='')
      elif opt in ("-D", "--d2"):
         if arg != "TAB":
           delimiter2 = arg
         print ('=> Delimiter for file 2 is "',delimiter2,'"',sep='')
      elif opt in ("-e", "--id"):
         identifier = arg
         print ('=> Identifier is "',identifier,'"',sep='')
      elif opt in ("-f", "--dout"):
         if arg == "TAB":
           out_delimiter = '\t'
         else:
           out_delimiter = arg
         print ('=> Output delimiter is "',out_delimiter,'"',sep='')
      elif opt in ("-c", "--cs"):
         casesensitive = True
         print ('=> Case sensitive is "',casesensitive,'"',sep='')
      elif opt in ("-S", "--subst"):
         substitute = True
         print ('=> Substitute missing values with zeroes is "',substitute,'"',sep='')
      elif opt in ("-n", "--nthr"):
         try:
            norm_threshold = float(arg)
         except ValueError:
            print (' Threshold value "'+arg+'" is not a number!')
            norm_threshold = 0.0
         normalize = True
         print ('=> Normalization threshold is "',norm_threshold,'"',sep='')
         
   if (not casesensitive):
       identifier = identifier.upper()

   print ('Input file 1 is "',inputfile1,'"',sep='')
   print ('Input file 2 is "',inputfile2,'"',sep='')
   print ('Output file 1 is "',outputfile1,'"',sep='')
   print ('Output file 2 is "',outputfile2,'"',sep='')
   print ('Delimiter for file 1 is "',delimiter1,'"',sep='')
   print ('Delimiter for file 2 is "',delimiter2,'"',sep='')
   print ('Identifier is "',identifier,'"',sep='')
   if (out_delimiter != default_out_delimiter_value):
     print ('Output delimiter is "',out_delimiter,'"',sep='')
   print ('Case sensitive is "',casesensitive,'"',sep='')
   print ('Substitute missing values with zeroes is "',substitute,'"',sep='')
   print ('Normalize is "',normalize,'" (threshold is "',norm_threshold,'")',sep='')
    
   with open(inputfile1, newline='') as csvinfile1:
     if (casesensitive):  
         reader1 = csv.DictReader(csvinfile1, delimiter=delimiter1)
     else:
         reader1 = csv.DictReader((l.upper() for l in csvinfile1), delimiter=delimiter1)
     fn1 = reader1.fieldnames
     len1 = len(fn1)
  #   print(fn1)
     if (identifier not in fn1):
       message = "Identifier \""+identifier+"\" is not present in the file \""+inputfile1+"\" header."
       print(message)
       sys.exit(message)

     with open(inputfile2, newline='') as csvinfile2:
       if (casesensitive):  
         reader2 = csv.DictReader(csvinfile2, delimiter=delimiter2)
       else:
         reader2 = csv.DictReader((l.upper() for l in csvinfile2), delimiter=delimiter2)  
       
       fn2 = reader2.fieldnames
       len2 = len(fn2)
   #    print(fn2)
       if (identifier not in fn2):
         message = "Identifier \""+identifier+"\" is not present in the file \""+inputfile2+"\" header."
         print(message)
         sys.exit(message)

# Get column names common for the both files
       colnames = [identifier]
       for name in fn1:
         if (name == identifier):
         # do nothing
           True
         elif (name in fn2):
           colnames.append(name)
         else:
           print ("Column \""+name+"\" is not present in the file \""+inputfile2+"\" header.")
         
       for name in fn2:
         if (name not in colnames):
           print ("Column \""+name+"\" is not present in the file \""+inputfile1+"\" header.")
         
       colnames[1:] = sorted(colnames[1:])     
       print("Collected and sorted column names:",colnames)

       if (normalize or substitute):       
         numcolnames = colnames[1:]
         print("Potentially numerical columns:",numcolnames)
       
       # Do not include duplicate identifiers or identifiers not presented in both files
       ids_not = set()
       ids1 = set()
       for row in reader1:
         #if (not casesensitive):
          #   row = row.upper();
 #        print(row.keys())
         if (row[identifier] not in ids_not) :
             if (row[identifier] not in ids1) :
                 ids1.add(row[identifier])  
                 if (normalize or substitute):
                    for name in numcolnames:
                        try:
                            if is_blank(row[name]):
                                row[name] = 0
                            val = float(row[name])
                        except ValueError:
                            print("Value ("+name+") "+row[name]+" is not a number!")
                            numcolnames.remove(name)
             else :
                 ids_not.add(row[identifier])
                 ids1.remove(row[identifier])
                 
#       print(ids1)          
         
       ids2 = set()
       for row in reader2:
#         if (not casesensitive):
 #            row = row.upper();    
         if (row[identifier] not in ids_not) :
             if (row[identifier] not in ids2) :  
                 ids2.add(row[identifier])
                 if (normalize or substitute):
                    for name in numcolnames:
                        try:
                            if is_blank(row[name]):
                                row[name] = 0
                            val = float(row[name])
                        except ValueError:
                            print("Value ("+name+") "+row[name]+" is not a number!")
                            numcolnames.remove(name)
             else :
                 ids_not.add(row[identifier])
                 ids2.remove(row[identifier])
                 
#       print(ids2)          
                 
       ids1.intersection_update(ids2)       
       
#       print(ids1)

       if (normalize or substitute) :       
         print("Final list of numerical columns:",numcolnames)
         
       csvinfile1.seek(0)
       csvinfile2.seek(0)       
       
       if (not casesensitive):  
         reader1 = csv.DictReader((l.upper() for l in csvinfile1), delimiter=delimiter1)
         reader2 = csv.DictReader((l.upper() for l in csvinfile2), delimiter=delimiter2)
       
       
       # Create dialects similar to "excel"
       dialect1 = csv.excel()
       if (out_delimiter == default_out_delimiter_value):
         dialect1.delimiter = delimiter1
       else:
         dialect1.delimiter = out_delimiter
         
       dialect2 = csv.excel()
       if (out_delimiter == default_out_delimiter_value):
         dialect2.delimiter = delimiter2
       else:
         dialect2.delimiter = out_delimiter
         
       errmax = 5
       errcount = errmax
       
#       print("===========")
       
       non_zero_sum1 = 0.0
       non_zero_count1 = 0
       
       dict1 = {}
       for row in reader1:
#           print(row.keys())
           if (len(row) > len1):
             message = "Data row length exceeds length of the header row in \""+inputfile1+"\"."
             print(message)
             sys.exit(message)
           elif (row[identifier] in ids1):
             if (normalize or substitute):
                for name in numcolnames:
                    try:
                        flo = float(row[name])
                        if (flo >= norm_threshold):
                            non_zero_sum1 += flo
                            non_zero_count1 += 1
                        else:
                            row[name] = 0.0
                    except ValueError:
#                        print("Value ("+name+") "+row[name]+" is not a number! (Dict1)")
#                        print(row)
                         row[name] = 0.0
                
             dict1[row[identifier]] = row

           else:
             if (row[identifier] == identifier):
             # do nothing - it's header
               True
             elif (errcount > 0):
               print("Identifier \""+row[identifier]+"\" appears more than once in \""+inputfile1+"\" or is not present in \""+inputfile2+"\".")
               errcount -= 1
             elif (errcount == 0):
               print("... more warnings...")
               errcount -= 1
               
#       print("===========")        
#       print(dict1)  
       if (normalize):
           print("Non_zero_sum1 = "+str(non_zero_sum1)+"   non_zero_count1 = "+str(non_zero_count1))
#       print("===========")

       non_zero_sum2 = 0.0
       non_zero_count2 = 0
               
       dict2 = {}
       for row in reader2:
           if (len(row) > len2):
             message = "Data row length exceeds length of the header row in \""+inputfile2+"\"."
             print(message)
             sys.exit(message)
           elif (row[identifier] in ids1):  #ids1 is the correct common collection!
             if (normalize or substitute):
                for name in numcolnames:
                    try:
                        flo = float(row[name])
                        if (flo > 0.0):
                            non_zero_sum2 += flo
                            non_zero_count2 += 1
                    except ValueError:
#                        print("Value ("+name+") "+row[name]+" is not a number! (Dict2)")
#                        print(row)
                         row[name] = 0.0
                        
             dict2[row[identifier]] = row
             
           else:
             if (row[identifier] == identifier):
             # do nothing - it's header
               True
             elif (errcount > 0):
               print("Identifier \""+row[identifier]+"\" appears more than once in \""+inputfile2+"\" or is not present in \""+inputfile1+"\".")
               errcount -= 1
             elif (errcount == 0):
               print("... more warnings...")
               errcount -= 1
       
#       print("===========")
#       print(dict2) 
       if (normalize):
           print("Non_zero_sum2 = "+str(non_zero_sum2)+"   non_zero_count2 = "+str(non_zero_count2))
           if (non_zero_count1 == 0) or (non_zero_count2 == 0):
                n_coeff = 1.0
           else:
                n_coeff = (non_zero_sum1 / non_zero_count1) / (non_zero_sum2 / non_zero_count2)
           print("Normalisation coefficient = "+str(n_coeff))
           
#       N-O-R-M-A-L-I-S-A-T-I-O-N 

           for key in dict2:
             row = dict2[key]
#             print("dic2["+key+"]:"+str(row))
             for name in numcolnames:
#                print("row["+name+"]="+str(row[name]))
                row[name] = n_coeff * float(row[name])
             dict2[key] = row
           
           
#       print("===========")
       
       with open(outputfile1, 'w', newline='') as csvoutfile1:
            writer1 = csv.DictWriter(csvoutfile1, fieldnames=colnames, restval='', extrasaction='ignore', dialect=dialect1)
            writer1.writeheader()
            for sortedKey in sorted(dict1) :
                writer1.writerow(dict1[sortedKey])
             
       with open(outputfile2, 'w', newline='') as csvoutfile2:
            writer2 = csv.DictWriter(csvoutfile2, fieldnames=colnames, restval='', extrasaction='ignore', dialect=dialect2)
            writer2.writeheader()
            for sortedKey in sorted(dict2) :
                writer2.writerow(dict2[sortedKey])
         

if __name__ == "__main__":
   main(sys.argv[1:])