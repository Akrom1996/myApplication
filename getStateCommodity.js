'use strict';

const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');

const ccpPath = path.resolve(__dirname, '..', '..', 'first-network', 'connection-org1.json');

const channelName = "mychannel";
const smartContractName = "contract";

async function main(traderId) {
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
        console.log("\nGet current state of commodities"); 
        const getState = await contract.evaluateTransaction("getStateComm");
        

        // Disconnect from the gateway.
        await gateway.disconnect();
        //console.log(getState.toString());
        return getState.toString();

    } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}

//main();
module.exports.main = main;