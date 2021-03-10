'use strict';

const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');
const csvjson = require('csvjson');
const writeFile = require('fs').writeFile;

const ccpPath = path.resolve(__dirname, '..', '..', 'first-network', 'connection-org1.json');

const channelName = "mychannel";
const smartContractName = "contract";

async function main(assetName, userName) {
    try {
        const appAdmin = userName;
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
        //console.log("\nGet current state of commodities");
        let time = [];

        for (let i = 1; i <= 10; i++) {
            let starTime = new Date().getTime();
            assetName = "nodeA_asset" + i.toString();
            const getState = await contract.submitTransaction("commodityTrade", assetName, appAdmin);
            let endTime = new Date().getTime()
            let sec = (((endTime - starTime) % 60000) / 1000).toFixed(2);
            console.log(JSON.parse(getState.toString()))
            console.log(assetName)            
            time.push({ seconds: sec.toString() })

        }
        console.log("\n")
        console.log(time)
        const csvData = csvjson.toCSV((time), {
            headers: 'key'
        });
        writeFile('./time-data.csv', csvData, (err) => {
            if (err) {
                console.log(err); // Do something to handle the error or just throw it
                throw new Error(err);
            }
            console.log('Success!');
        });

        /*const getState = await contract.submitTransaction("commodityTrade", assetName, appAdmin);
        console.log(JSON.parse(getState.toString()))*/
        //console.log(getState.toString())
        // Disconnect from the gateway.
        await gateway.disconnect();

        //return JSON.parse(getState.toString());JSON.parse(getState.toString())

    } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}

main("nodeA_asset1", "nodeB");
//module.exports.main = main;