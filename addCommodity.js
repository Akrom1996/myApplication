'use strict';

const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');

const ccpPath = path.resolve(__dirname, '..', '..', 'first-network', 'connection-org1.json');
const channelName = "mychannel";
const smartContractName = "contract";

async function main(tradingSymbol, tradingEnergyPrice, traderId, Amount) {
    try {
        const appAdmin = traderId;
        console.log(tradingSymbol)
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
       console.log('\nSubmit AddCommodity transaction.');
       const time = new Date().toString().split('G')[0].trim()
       const result = await contract.submitTransaction('addCommodity', tradingSymbol, tradingEnergyPrice, traderId, Amount, time);
       console.log(JSON.parse(result.toString()));
       /*console.log('addCommodityResponse: ');
       console.log(addCommodityResponse.toString('utf8'));
       console.log('addCommodityResponse_JSON.parse: ');
       console.log(JSON.parse(addCommodityResponse.toString()));
*/
        await gateway.disconnect();
        return {message: "Energy was successfully added"};
        
    } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}
// main("_asset13", "621", "A", Number((Math.random() * 100).toFixed(2)).toString(), "2019-01-01")

module.exports.main = main;