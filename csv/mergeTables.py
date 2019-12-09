import csv

with open('AML_sample_transactions.csv') as transactionsCSV:
    with open('AML_sample_alert_scored.csv') as alertsCSV:
        with open('AML_combined.csv', mode='w') as combined:
            combine = csv.writer(combined, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            transactionsRead = csv.reader(transactionsCSV, delimiter=',')
            transactions = []
            for row in transactionsRead:
                transactions.append(row)
            alertsRead = csv.reader(alertsCSV, delimiter=',')
            alerts = []
            for row in alertsRead:
                alerts.append(row)
            for i in range(len(transactions)):
                row = transactions[i]
                for j in range(len(alerts)):
                    row2 = alerts[j]
                    if (i == 0 and j == 0) or (row[5] == row2[3] and row[6] == row2[4] and row[1] == row2[5]):
                        newRow = row
                        newRow2 = row2
                        newRow2.remove(newRow2[3])
                        newRow2.remove(newRow2[4])
                        newRow2.remove(newRow2[5])
                        newRow.extend(newRow2)
                        combine.writerow(newRow)
                        break
                    

