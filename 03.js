'use strict';

const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');
const fs = require('fs');

const ccpPath = path.resolve(__dirname, '..', '..', 'first-network', 'connection-org1.json');
const channelName = "mychannel";
const smartContractName = "contract";

// async function main(tradingSymbol, traderId, dataArray) {
async function main(tradingSymbol, energyPricePerKw,traderId, amount, date) {
    try {
        const appAdmin = traderId;
        // Create a new file system based wallet for managing identities.
        const walletPath = path.join('/home/akrom/fabric-samples/myBlockchain/application', 'wallet');
        const wallet = new FileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check to see if we've already enrolled the user.
        const userExists = await wallet.exists(appAdmin);
        if (!userExists) {
            console.log(`An identity for the user ${appAdmin} does not exist in the wallet`);
            console.log('Run the enrollAdmin.js application before retrying');
            return;
        }

        // Create a new gateway for connecting to our peer node.
        const gateway = new Gateway();
        await gateway.connect(ccpPath, { wallet, identity: appAdmin, discovery: { enabled: true, asLocalhost: true } });
        // Get the network (channel) our contract is deployed to.
        const network = await gateway.getNetwork(channelName);

        // Get the contract from the network.
        const contract = network.getContract(smartContractName);


        //Submit create a commodity transaction
        /*console.log('\nSubmit AddCommodity transaction.');
        const response = await contract.submitTransaction('addCommodity', tradingSymbol,energyPricePerKw,traderId,amount,date)
        console.log(JSON.parse(response.toString()));*/
        for (let i = 1; i <= 1; i++) {
            let energyPricePerKw = "750";//Number(((Math.random() * 10 +  750).toFixed(2))).toString();
            console.log('\nSubmit AddCommodity transaction.');
            const addCommodityResponse = await contract.submitTransaction('addCommodity', tradingSymbol + i.toString(), energyPricePerKw,  traderId, amount, date);
            console.log(JSON.parse(addCommodityResponse.toString()));
        }
        /*console.log('addCommodityResponse: ');
        console.log(addCommodityResponse.toString('utf8'));
        console.log('addCommodityResponse_JSON.parse: ');
        console.log(JSON.parse(addCommodityResponse.toString()));
 */
        await gateway.disconnect();


    } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}
// var dataArray;
// fs.readFile('nodeData.csv', 'utf8', function (err, data) {
//     dataArray = data.split(/\r?\n/);

//     //console.log(dataArray[i].split(',')[0]);
//     main("nodeA_asset", "nodeA", dataArray)
// })
/*for (let i = 1; i < 20; i++)dataArray[i].split(',')[2].toString()
dataArray[i].split(',')[0].toString()
    */
   
//module.exports.main =  main;

main("_asset12", "621", "A", Number((Math.random() * 100).toFixed(2)).toString(), "2019-01-01")
