const nodeA_add_asset = require('./03');
const check_asset = require('./04');
const updateBalance = require('./05');
const buyEnergy = require('./06');

const path = require('path');
const fs = require('fs');


var dataArray;
fs.readFile('nodeData.csv', 'utf8', function (err, data) {
    dataArray = data.split(/\r?\n/);
    //console.log(dataArray[i].split(',')[0]);    
    let date = dataArray[2].split(',')[0].toString();
    //console.log(date)
    try {
        nodeA_add_asset.main("nodeA_" + date, "750", "nodeA", Math.abs(dataArray[2].split(',')[2]).toString(), date);
    }
    catch (err) {
        console.log(err)
    }
    try {
        updateBalance.main("nodeB")
    }
    catch (err) {
        console.log(err)
    }
    let assets;
    try {
        assets = check_asset.main("nodeA");
    }
    catch (err) {
        console.log(err)
    }
    if (assets.toString().length > 0) {
        //buy Energy
        buyEnergy.main("nodeA" + date, "nodeB")
    }
})

/*
let date = dataArray[0].split(',')[0].toString();
nodeA_add_asset.main("nodeA_" + date, "750", "nodeA", Math.abs(dataArray[0].split(',')[2]).toString(), date);
updateBalance.main("nodeB")
let assets = check_asset.main("nodeA");
if(assets.toString().length > 0){
    //buy Energy
    buyEnergy.main("nodeA"+date, "nodeB")
}
*/
