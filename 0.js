/*const csv = require('csv-parser');
const fs = require('fs');
function main(){
fs.createReadStream('nodeData.csv')
    .pipe(csv())
    .on('data', (row) => {
        //console.log(row.date)
        return [row.date, row.Elec_kW]
    })
}
module.exports = main;*/
/*
for (let i = 0; i < DateTime.length; i++) {
    console.log(DateTime.shift(), electricity.shift())
}*/
/*
const fs = require('fs');
var dataArray;
fs.readFile('nodeData.csv', 'utf8', function (err, data) {
  dataArray = data.split(/\r?\n/);
  for (let i = 0; i < dataArray.length; i++) {
    console.log(dataArray[i].split(',')[2]);
}
 
});*/

/*for (let i = 0; i < dataArray.length; i++) {
    console.log(dataArray[i])
}*/

let date = new Date().toString().split("G")[0];
console.log(date)
let amount = parseInt(Math.random() * 20) + 5;
console.log(amount)
let tx = 0;
for(let i = 0; i < 10; i ++){
    tx ++;
    console.log(tx)
}